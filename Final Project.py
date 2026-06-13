import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import requests
import os
import math
import time
import folium
from datetime import datetime, timedelta
import polyline
import concurrent.futures

# PARAMETER INITIALIZATION
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
MAX_ITER_ALL = 150

N_RUNS = 1  # <--- DIUBAH MENJADI 1 AGAR RUNNING CEPAT SAAT TESTING

# DATA SOURCE & BOUNDARIES
file_path = 'Data Kurir Paket Yudi Hiparni.csv' 
DC_JUANDA = (-7.3817573, 112.7544115)
route_cache = {}

def get_distance_matrix(coords):
    if len(coords) <= 100:
        coord_str = ";".join([f"{c[1]},{c[0]}" for c in coords])
        url = f"http://router.project-osrm.org/table/v1/driving/{coord_str}?sources=all&destinations=all&annotations=distance"
        try:
            r = requests.get(url, timeout=30).json()
            if r.get('code') == 'Ok': return np.array(r['distances']) / 1000
        except: pass

    # Fallback: Haversine
    R = 6371.0
    mat = np.zeros((len(coords), len(coords)))
    for i in range(len(coords)):
        for j in range(len(coords)):
            if i != j:
                lat1, lon1 = math.radians(coords[i][0]), math.radians(coords[i][1])
                lat2, lon2 = math.radians(coords[j][0]), math.radians(coords[j][1])
                a = math.sin((lat2 - lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1)/2)**2
                mat[i][j] = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return mat

def get_osrm_route_cached(p1, p2):
    cache_key = (round(p1[0],5), round(p1[1],5), round(p2[0],5), round(p2[1],5))
    if cache_key in route_cache: return route_cache[cache_key]
    url = f"http://router.project-osrm.org/route/v1/driving/{p1[1]},{p1[0]};{p2[1]},{p2[0]}?overview=full&geometries=polyline"
    try:
        r = requests.get(url, timeout=10).json()
        if r['code'] == 'Ok':
            geom = polyline.decode(r['routes'][0]['geometry'])
            route_cache[cache_key] = (geom, r['routes'][0]['distance']/1000, r['routes'][0]['duration']/3600)
            return route_cache[cache_key]
    except: pass
    route_cache[cache_key] = ([p1, p2], math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)*111, 0.05)
    return route_cache[cache_key]

# --- DEMAND AGGREGATION & PREPROCESSING ---
def load_and_group_data(path):
    try: data = pd.read_csv(path, sep=';', engine='python', encoding='utf-8-sig')
    except: data = pd.read_csv(path, sep=',', engine='python', encoding='utf-8-sig')
    data.columns = [str(c).strip() for c in data.columns]

    if 'Koordinat' in data.columns:
        data = data.dropna(subset=['Nosi', 'Koordinat']).copy()
        data['lat'] = data['Koordinat'].apply(lambda x: float(x.split(',')[0]) if pd.notna(x) else None)
        data['lon'] = data['Koordinat'].apply(lambda x: float(x.split(',')[1]) if pd.notna(x) else None)
        data['AWB'] = data['Nosi']
        if 'Produk' in data.columns: data['Produk'] = data['Produk'].fillna('PKH')
        else: data['Produk'] = 'PKH'

    elif 'Latitude' in data.columns:
        data = data.dropna(subset=['Latitude', 'Longitude']).copy()
        data['lat'] = data['Latitude'].apply(lambda x: float(str(x).replace(',','.')))
        data['lon'] = data['Longitude'].apply(lambda x: float(str(x).replace(',','.')))
        if 'AWB' not in data.columns: data['AWB'] = data.index
        if 'Produk' not in data.columns: data['Produk'] = 'PKH'

    data = data.dropna(subset=['lat', 'lon']).reset_index(drop=True).head(141)
    data['service_time'] = data['Produk'].apply(lambda x: 3 if str(x).upper().strip() == 'PE' else 2)
    data['is_priority'] = data['Produk'].apply(lambda x: 1 if str(x).upper().strip() == 'PE' else 0)

    df_grouped = data.groupby(['lat', 'lon']).agg({
        'AWB': lambda x: list(x), 'Produk': lambda x: list(x),
        'service_time': 'sum', 'is_priority': 'max'   
    }).reset_index()

    df_grouped['jumlah_paket'] = df_grouped['AWB'].apply(len)
    df_grouped['coords'] = list(zip(df_grouped['lat'], df_grouped['lon']))
    return df_grouped

# --- COST FUNCTION & SLA EVALUATION ---
SPEED_KMPM = 40 / 60
SLA_LIMIT_PE = 360

def calc_cost(route, matrix, node_data):
    total_dist, current_time, total_penalty, curr_node = 0, 0, 0, 0
    for next_node in route:
        dist = matrix[curr_node][next_node]
        total_dist += dist
        current_time += (dist / SPEED_KMPM)
        current_time += node_data[next_node]['service_time']
        if node_data[next_node]['is_priority'] == 1 and current_time > SLA_LIMIT_PE:
            total_penalty += 1000 * (current_time - SLA_LIMIT_PE)
        curr_node = next_node
    total_dist += matrix[curr_node][0]
    return total_dist + total_penalty

# --- METAHEURISTICS ALGORITHMS (Di-upgrade dengan Parameter Dinamis) ---

# SA ditambahkan parameter cooling dan temp
def optimize_SA(num_p, matrix, node_data, cooling=0.9176, temp=100.0):
    curr_r = list(range(1, num_p + 1)); random.shuffle(curr_r)
    curr_d = calc_cost(curr_r, matrix, node_data)
    best_r, best_d, history = copy.deepcopy(curr_r), curr_d, []

    for _ in range(MAX_ITER_ALL):
        new_r = copy.deepcopy(curr_r); i, j = random.sample(range(num_p), 2)
        new_r[i], new_r[j] = new_r[j], new_r[i]
        new_d = calc_cost(new_r, matrix, node_data)
        if new_d < curr_d or random.random() < math.exp(max(-100, (curr_d - new_d) / temp)):
            curr_r, curr_d = new_r, new_d
            if curr_d < best_d: best_r, best_d = copy.deepcopy(curr_r), curr_d
        history.append(best_d); temp *= cooling
    return best_r, best_d, history

def optimize_ACO(num_p, matrix, node_data):
    n_ants, alpha, beta, evap = 25, 0.67, 2.09, 0.23
    pheromone = np.ones((num_p+1, num_p+1))
    best_r, best_d, history = None, float('inf'), []

    for _ in range(MAX_ITER_ALL):
        for _ in range(n_ants):
            unvisited = list(range(1, num_p+1)); curr, route = 0, []
            while unvisited:
                probs = [(pheromone[curr][n]**alpha) * ((1.0/(matrix[curr][n]+0.001))**beta) for n in unvisited]
                probs = np.array(probs) / sum(probs)
                next_node = np.random.choice(unvisited, p=probs)
                route.append(next_node); unvisited.remove(next_node); curr = next_node
            total_cost = calc_cost(route, matrix, node_data)
            if total_cost < best_d: best_r, best_d = route, total_cost
            curr_n = 0
            for node in route: pheromone[curr_n][node] += 1.0 / total_cost; curr_n = node
        pheromone *= (1 - evap); history.append(best_d)
    return best_r, best_d, history

# GA ditambahkan parameter pop_size dan mut_rate
def optimize_GA(num_p, matrix, node_data, pop_size=55, mut_rate=0.32):
    pop = [random.sample(range(1, num_p + 1), num_p) for _ in range(pop_size)]
    best_r, best_d, history = None, float('inf'), []

    for _ in range(MAX_ITER_ALL):
        pop = sorted(pop, key=lambda x: calc_cost(x, matrix, node_data))
        current_best_cost = calc_cost(pop[0], matrix, node_data)
        if current_best_cost < best_d: best_r, best_d = copy.deepcopy(pop[0]), current_best_cost
        history.append(best_d)

        next_gen = pop[:10]
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(pop[:20], 2); pt = random.randint(1, num_p-1)
            child = p1[:pt] + [g for g in p2 if g not in p1[:pt]]
            if random.random() < mut_rate:
                i, j = random.sample(range(num_p), 2); child[i], child[j] = child[j], child[i]
            next_gen.append(child)
        pop = next_gen
    return best_r, best_d, history

def optimize_LBA(num_p, matrix, node_data):
    pop_size, max_loop = 69, 20
    pop = [random.sample(range(1, num_p + 1), num_p) for _ in range(pop_size)]
    g_best_r, g_best_d, history = None, float('inf'), []

    for _ in range(MAX_ITER_ALL):
        fitnesses = [calc_cost(ind, matrix, node_data) for ind in pop]
        c_best_idx = np.argmin(fitnesses)
        c_best_r, c_best_d = copy.deepcopy(pop[c_best_idx]), fitnesses[c_best_idx]
        if c_best_d < g_best_d: g_best_r, g_best_d = copy.deepcopy(c_best_r), c_best_d
        else:
            ls_r, ls_d = copy.deepcopy(g_best_r), g_best_d
            for _ in range(max_loop):
                temp_r = copy.deepcopy(ls_r); i, j = random.sample(range(num_p), 2)
                temp_r[i], temp_r[j] = temp_r[j], temp_r[i]
                temp_d = calc_cost(temp_r, matrix, node_data)
                if temp_d < ls_d: ls_r, ls_d = temp_r, temp_d
            if ls_d < g_best_d: g_best_r, g_best_d = copy.deepcopy(ls_r), ls_d
        history.append(g_best_d)

        new_pop = [copy.deepcopy(g_best_r)]
        while len(new_pop) < pop_size:
            S = random.choice(pop[:20])
            if random.random() < 0.9:
                idx1, idx2 = sorted(random.sample(range(num_p), 2)); a = random.randint(1, 5); cand = copy.deepcopy(S)
                if a == 1 and idx2 < num_p - 1: cand[idx1:idx1+2], cand[idx2:idx2+2] = cand[idx2:idx2+2], cand[idx1:idx1+2]
                elif a == 2: cand[idx1:idx2+1] = reversed(cand[idx1:idx2+1])
                elif a == 3: cand[idx1], cand[idx2] = cand[idx2], cand[idx1]
                elif a == 4: val = cand.pop(idx1); cand.insert(idx2, val)
                elif a == 5: sub = cand[idx1:idx2+1]; random.shuffle(sub); cand[idx1:idx2+1] = sub
            else: cand = copy.deepcopy(S); random.shuffle(cand)
            new_pop.append(cand)
        pop = new_pop
    return g_best_r, g_best_d, history

# --- PARALLEL RUNNER (Di-upgrade menerima **kwargs parameter) ---
def single_run(algo_func, num_p, matrix, node_data, **kwargs):
    t0 = time.time()
    r, c, h = algo_func(num_p, matrix, node_data, **kwargs)
    t_elapsed = time.time() - t0
    return r, c, h, t_elapsed

def run_algo_10x_parallel(algo_func, num_p, matrix, node_data, **kwargs):
    costs, times = [], []
    best_overall_cost = float('inf')
    best_overall_route, best_overall_history = None, []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(single_run, algo_func, num_p, matrix, node_data, **kwargs) for _ in range(N_RUNS)]
        for future in concurrent.futures.as_completed(futures):
            r, c, h, t_elapsed = future.result()
            costs.append(c)
            times.append(t_elapsed)
            if c < best_overall_cost:
                best_overall_cost = c
                best_overall_route = r
                best_overall_history = h
                
    return best_overall_route, best_overall_cost, np.mean(costs), np.std(costs), np.mean(times), best_overall_history


# --- BLOK UTAMA (WAJIB DI WINDOWS) ---
if __name__ == '__main__':
    df_grouped = load_and_group_data(file_path)

    trip_ids, current_trip, current_load = [], 0, 0
    for idx, row in df_grouped.iterrows():
        if current_load + row['jumlah_paket'] > 50:
            current_trip += 1
            current_load = 0
        trip_ids.append(current_trip)
        current_load += row['jumlah_paket']
    df_grouped['trip_id'] = trip_ids

    print(f"[PREPROCESSING] Total: {df_grouped['jumlah_paket'].sum()} paket | {len(df_grouped)} titik lokasi unik.")
    for t in df_grouped['trip_id'].unique():
        n_paket = df_grouped[df_grouped['trip_id']==t]['jumlah_paket'].sum()
        n_titik = len(df_grouped[df_grouped['trip_id']==t])
        n_pe = df_grouped[(df_grouped['trip_id']==t) & (df_grouped['is_priority']==1)]['jumlah_paket'].count()
        print(f" - Trip {t+1}: {n_paket} paket | {n_titik} titik antar | {n_pe} titik prioritas PE.")

    komparasi_results = []
    all_routes_map = {'SA': [], 'ACO': [], 'GA': [], 'LBA': []}
    history_plot = {'SA': [], 'ACO': [], 'GA': [], 'LBA': []}

    print(f"\n[PROSES] Memulai iterasi {N_RUNS}x Paralel dengan Auto-Tuning Hyperparameter...")

    for i in range(len(df_grouped['trip_id'].unique())):
        df_segment = df_grouped[df_grouped['trip_id'] == i].copy()
        coords_trip = [DC_JUANDA] + df_segment['coords'].tolist()
        matrix = get_distance_matrix(coords_trip)
        num_p = len(df_segment)
        total_paket_trip = df_segment['jumlah_paket'].sum()
        total_pe_trip = df_segment[df_segment['is_priority'] == 1]['jumlah_paket'].count()

        print(f"\n -> Memproses Trip {i+1} ({num_p} titik, {total_paket_trip} paket)")
        node_data = {idx_nd + 1: {'service_time': row_nd['service_time'], 'is_priority': row_nd['is_priority']} 
                     for idx_nd, row_nd in df_segment.reset_index(drop=True).iterrows()}

        # -------------------------------------------------------------
        # FITUR BARU: GRID SEARCH HYPERPARAMETER TUNING OTOMATIS
        # -------------------------------------------------------------
        print("    [*] Melakukan Tuning Parameter SA & GA...")
        
        # 1. Tuning SA (Mencari Kombinasi Cooling & Temp Terbaik)
        best_sa_cost, best_sa_params = float('inf'), {'cooling': 0.91, 'temp': 100.0}
        for c_val in [0.85, 0.91, 0.98]:       # Uji 3 cooling rate
            for t_val in [50.0, 100.0, 200.0]: # Uji 3 suhu awal
                _, cost, _ = optimize_SA(num_p, matrix, node_data, cooling=c_val, temp=t_val)
                if cost < best_sa_cost:
                    best_sa_cost, best_sa_params = cost, {'cooling': c_val, 'temp': t_val}

        # 2. Tuning GA (Mencari Kombinasi Pop Size & Mutation Rate Terbaik)
        best_ga_cost, best_ga_params = float('inf'), {'pop_size': 50, 'mut_rate': 0.3}
        for p_size in [30, 50, 80]:         # Uji 3 ukuran populasi
            for m_rate in [0.1, 0.3, 0.5]:  # Uji 3 probabilitas mutasi
                _, cost, _ = optimize_GA(num_p, matrix, node_data, pop_size=p_size, mut_rate=m_rate)
                if cost < best_ga_cost:
                    best_ga_cost, best_ga_params = cost, {'pop_size': p_size, 'mut_rate': m_rate}
        
        print(f"    [+] SA Tuned Parameter -> Cooling: {best_sa_params['cooling']} | Temp: {best_sa_params['temp']}")
        print(f"    [+] GA Tuned Parameter -> Pop Size: {best_ga_params['pop_size']} | Mut Rate: {best_ga_params['mut_rate']}")
        # -------------------------------------------------------------

        # Eksekusi paralel menggunakan parameter TERBAIK hasil tuning
        print(f"    [*] Mengeksekusi Algoritma ({N_RUNS}x Run)...")
        best_r_sa, b_c_sa, a_c_sa, s_c_sa, t_sa, h_sa = run_algo_10x_parallel(
            optimize_SA, num_p, matrix, node_data, cooling=best_sa_params['cooling'], temp=best_sa_params['temp'])
        
        best_r_ga, b_c_ga, a_c_ga, s_c_ga, t_ga, h_ga = run_algo_10x_parallel(
            optimize_GA, num_p, matrix, node_data, pop_size=best_ga_params['pop_size'], mut_rate=best_ga_params['mut_rate'])
        
        # ACO dan LBA menggunakan default parameter
        best_r_aco, b_c_aco, a_c_aco, s_c_aco, t_aco, h_aco = run_algo_10x_parallel(optimize_ACO, num_p, matrix, node_data)
        best_r_lba, b_c_lba, a_c_lba, s_c_lba, t_lba, h_lba = run_algo_10x_parallel(optimize_LBA, num_p, matrix, node_data)

        if i == 0: 
            history_plot['SA'], history_plot['ACO'] = h_sa, h_aco
            history_plot['GA'], history_plot['LBA'] = h_ga, h_lba

        def process_route_data(algo_name, best_idx):
            r_coords = [DC_JUANDA] + [df_segment.iloc[idx-1]['coords'] for idx in best_idx] + [DC_JUANDA]
            full_geom, current_time_min, total_dist_km, sla_violations = [], 0, 0, 0
            paket_urutan = []

            for j in range(len(r_coords)-1):
                geom, d_km, d_hr = get_osrm_route_cached(r_coords[j], r_coords[j+1])
                if geom: full_geom.extend(geom)
                total_dist_km += d_km
                current_time_min += (d_hr * 60)

                if j < len(best_idx):
                    node_id = best_idx[j]
                    current_time_min += node_data[node_id]['service_time']
                    if node_data[node_id]['is_priority'] == 1 and current_time_min > SLA_LIMIT_PE:
                        sla_violations += 1
                    paket_urutan.append({
                        'urutan': j+1, 'koordinat': r_coords[j+1],
                        'waktu_sampai': current_time_min, 'is_pe': node_data[node_id]['is_priority']
                    })
            all_routes_map[algo_name].append({'geom': full_geom, 'paket': paket_urutan})
            return total_dist_km, (current_time_min / 60), sla_violations

        dist_sa, jam_sa, viol_sa = process_route_data('SA', best_r_sa)
        dist_aco, jam_aco, viol_aco = process_process_route_data = process_route_data('ACO', best_r_aco)
        dist_ga, jam_ga, viol_ga = process_route_data('GA', best_r_ga)
        dist_lba, jam_lba, viol_lba = process_route_data('LBA', best_r_lba)

        komparasi_results.append({
            'Trip': f'Trip {i+1}',
            'SA_Best': b_c_sa, 'SA_Avg': a_c_sa, 'SA_Std': s_c_sa, 'SA_Time(s)': t_sa, 'SA_Viol': viol_sa,
            'ACO_Best': b_c_aco, 'ACO_Avg': a_c_aco, 'ACO_Std': s_c_aco, 'ACO_Time(s)': t_aco, 'ACO_Viol': viol_aco,
            'GA_Best': b_c_ga, 'GA_Avg': a_c_ga, 'GA_Std': s_c_ga, 'GA_Time(s)': t_ga, 'GA_Viol': viol_ga,
            'LBA_Best': b_c_lba, 'LBA_Avg': a_c_lba, 'LBA_Std': s_c_lba, 'LBA_Time(s)': t_lba, 'LBA_Viol': viol_lba
        })

    # Cetak & Export
    df_hasil = pd.DataFrame(komparasi_results)
    print("\n[HASIL] Tabel Perbandingan Performa:")
    pd.set_option('display.max_columns', None) 
    print(df_hasil.to_string(index=False))
    
    df_hasil.to_csv('Hasil_Komparasi_Algoritma.csv', index=False)
    print("\n[INFO] Data hasil komparasi telah disimpan ke 'Hasil_Komparasi_Algoritma.csv'")

    # Grafik Matplotlib
    algos = ['SA', 'ACO', 'GA', 'LBA']
    avg_best = [df_hasil[f'{alg}_Best'].mean() for alg in algos]
    avg_avg = [df_hasil[f'{alg}_Avg'].mean() for alg in algos]
    avg_time = [df_hasil[f'{alg}_Time(s)'].mean() for alg in algos]

    x = np.arange(len(algos))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 6))

    bar1 = ax1.bar(x - width/2, avg_best, width, label='Rata-rata Best Cost', color='skyblue')
    bar2 = ax1.bar(x + width/2, avg_avg, width, label='Rata-rata Average Cost', color='royalblue')

    ax1.set_ylabel('Cost Rute (Jarak + Penalti)', color='black')
    ax1.set_title(f'Komparasi Performa Algoritma ({N_RUNS}x Run Paralel per Trip)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    line1 = ax2.plot(x, avg_time, color='red', marker='o', linestyle='-', linewidth=2, label='Rata-rata Running Time (s)')
    ax2.set_ylabel('Running Time (Detik)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

    plt.tight_layout()
    plt.savefig('Grafik_Komparasi_Algoritma.png')

   # ==========================================
    # VISUALIZATION INTERACTIVE GEOSPATIAL MAP
    # ==========================================
    map_komparasi = folium.Map(location=[DC_JUANDA[0], DC_JUANDA[1]], zoom_start=13, tiles='CartoDB positron')

    folium.Marker(
        location=DC_JUANDA, 
        popup='<b>Depot DC Juanda</b>', 
        icon=folium.Icon(color='black', icon='home')
    ).add_to(map_komparasi)

    trip_colors = {0: 'blue', 1: 'darkgreen', 2: 'purple', 3: 'cadetblue'}

    for algo_name, trips_data in all_routes_map.items():
        # KEMBALI KE 1 LAYER PER ALGORITMA DI KANAN ATAS
        is_default_show = True if algo_name == 'LBA' else False
        layer_group = folium.FeatureGroup(name=f"Rute & Marker ({algo_name})", show=is_default_show)
        
        for trip_idx, trip_info in enumerate(trips_data):
            warna_jalur_trip = trip_colors.get(trip_idx, 'gray')
            geom_jalan = trip_info['geom']
            paket_markers = trip_info['paket']
            
            # Menggambar Garis Rute
            if geom_jalan:
                folium.PolyLine(
                    locations=geom_jalan, color=warna_jalur_trip, weight=5, opacity=0.7,
                    tooltip=f"{algo_name} - Jalur Trip {trip_idx+1}"
                ).add_to(layer_group)
                
            # Menggambar Marker dengan Nomor Urut
            for p in paket_markers:
                urutan = p['urutan']
                is_pe = p['is_pe'] == 1
                
                warna_marker = '#d9534f' if is_pe else '#f0ad4e'
                status_prio = 'Paket Ekspres (PE)' if is_pe else 'Paket Biasa (Reguler)'
                
                # Class khusus untuk ditangkap oleh JavaScript (Filter Kiri Bawah)
                trip_class = f"trip-{trip_idx}"
                type_class = "type-pe" if is_pe else "type-reg"
                
                html_marker = f"""
                <div class="custom-marker {trip_class} {type_class}" style="
                    background-color: {warna_marker}; color: white; border-radius: 50%;
                    width: 24px; height: 24px; display: flex; justify-content: center;
                    align-items: center; font-weight: bold; font-size: 11px;
                    border: 2px solid white; box-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                ">
                    {urutan}
                </div>
                """
                
                waktu_est_wib = datetime.strptime("08:00", "%H:%M") + timedelta(minutes=p['waktu_sampai'])
                waktu_str = waktu_est_wib.strftime("%H:%M WIB")
                
                popup_html = f"""
                <div style="font-family: Arial; font-size: 12px; width: 220px;">
                    <h4 style="margin:0 0 5px 0; color:{warna_jalur_trip};">{algo_name} - Trip {trip_idx+1}</h4>
                    <hr style="margin: 3px 0; border: 0.5px solid #ccc;">
                    <b>Urutan Kirim:</b> Ke-{urutan}<br>
                    <b>Estimasi Tiba:</b> <span style="color:green; font-weight:bold;">{waktu_str}</span><br>
                    <b>Kategori:</b> {status_prio}
                </div>
                """
                
                folium.Marker(
                    location=p['koordinat'], 
                    icon=folium.DivIcon(html=html_marker, icon_size=(24,24), icon_anchor=(12,12)),
                    tooltip=f"[{algo_name} T{trip_idx+1}] Urutan {urutan} ({waktu_str})",
                    popup=folium.Popup(popup_html, max_width=250)
                ).add_to(layer_group)
                
        layer_group.add_to(map_komparasi)

    # Panel Kanan Atas (Hanya untuk milih Algoritma)
    folium.LayerControl(collapsed=False).add_to(map_komparasi)

    # =================================================================
    # KOTAK FILTER KIRI BAWAH (HTML + JAVASCRIPT)
    # =================================================================
    filter_html = '''
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 200px; 
        background-color: white; border:2px solid grey; z-index:9999; font-size:13px;
        padding: 10px; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); font-family: Arial;
        ">
        <b style="font-size:14px;">Filter Tampilan Peta</b><br>
        <hr style="margin: 5px 0;">
        <input type="checkbox" id="chk-t0" checked onchange="updateFilters()"> 
        <label for="chk-t0" style="color:blue; font-weight:bold;">Trip 1 (Biru)</label><br>
        
        <input type="checkbox" id="chk-t1" checked onchange="updateFilters()"> 
        <label for="chk-t1" style="color:darkgreen; font-weight:bold;">Trip 2 (Hijau)</label><br>
        
        <input type="checkbox" id="chk-t2" checked onchange="updateFilters()"> 
        <label for="chk-t2" style="color:purple; font-weight:bold;">Trip 3 (Ungu)</label><br>
        
        <hr style="margin: 5px 0;">
        <input type="checkbox" id="chk-reg" checked onchange="updateFilters()"> 
        <label for="chk-reg">Paket Biasa <span style="color:#f0ad4e;">(●)</span></label><br>
        
        <input type="checkbox" id="chk-pe" checked onchange="updateFilters()"> 
        <label for="chk-pe">Paket Ekspres <span style="color:#d9534f;">(●)</span></label><br>
    </div>

    <script>
    function updateFilters() {
        var t0 = document.getElementById('chk-t0').checked;
        var t1 = document.getElementById('chk-t1').checked;
        var t2 = document.getElementById('chk-t2').checked;
        var reg = document.getElementById('chk-reg').checked;
        var pe = document.getElementById('chk-pe').checked;

        // Atur visibilitas Marker (Lingkaran Angka)
        var markers = document.querySelectorAll('.custom-marker');
        markers.forEach(function(m) {
            var is_t0 = m.classList.contains('trip-0');
            var is_t1 = m.classList.contains('trip-1');
            var is_t2 = m.classList.contains('trip-2');
            var is_reg = m.classList.contains('type-reg');
            var is_pe = m.classList.contains('type-pe');

            var showTrip = (is_t0 && t0) || (is_t1 && t1) || (is_t2 && t2);
            var showType = (is_reg && reg) || (is_pe && pe);

            if (m.parentNode) {
                m.parentNode.style.display = (showTrip && showType) ? 'block' : 'none';
            }
        });

        // Atur visibilitas Garis Rute (SVG Path)
        var paths = document.querySelectorAll('path.leaflet-interactive');
        paths.forEach(function(p) {
            var color = p.getAttribute('stroke');
            if (color === 'blue') { p.style.display = t0 ? 'block' : 'none'; }
            if (color === 'darkgreen') { p.style.display = t1 ? 'block' : 'none'; }
            if (color === 'purple') { p.style.display = t2 ? 'block' : 'none'; }
        });
    }

    // Eksekusi ulang jika user ganti algoritma di kotak kanan atas
    var observer = new MutationObserver(function() { updateFilters(); });
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Inisialisasi awal
    setTimeout(updateFilters, 500);
    </script>
    '''
    map_komparasi.get_root().html.add_child(folium.Element(filter_html))

    file_html_peta = 'Dashboard_Peta_Komparasi_EAS.html'
    map_komparasi.save(file_html_peta)

    print(f"\n[OUTPUT] Gambar Grafik tersimpan pada: 'Grafik_Komparasi_Algoritma.png'")
    print(f"[OUTPUT] File dashboard geospatial tersimpan pada: '{file_html_peta}'")