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

# PARAMETER INITIALIZATION
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
MAX_ITER_ALL = 150

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

    # Grouping lokasi koordinat sama
    df_grouped = data.groupby(['lat', 'lon']).agg({
        'AWB': lambda x: list(x),
        'Produk': lambda x: list(x),
        'service_time': 'sum', 
        'is_priority': 'max'   
    }).reset_index()

    df_grouped['jumlah_paket'] = df_grouped['AWB'].apply(len)
    df_grouped['coords'] = list(zip(df_grouped['lat'], df_grouped['lon']))
    df_grouped = df_grouped.sort_values(by='lon').reset_index(drop=True)
    return df_grouped

df_grouped = load_and_group_data(file_path)

# --- TRIP PARTITIONING (CAPACITY CONTRAINT: 50) ---
trip_ids = []
current_trip = 0
current_load = 0

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


# --- COST FUNCTION & SLA EVALUATION ---
SPEED_KMPM = 40 / 60
SLA_LIMIT_PE = 360

def calc_cost(route, matrix, node_data):
    total_dist = 0
    current_time = 0
    total_penalty = 0
    curr_node = 0

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


# --- METAHEURISTICS ALGORITHMS ---
def optimize_SA(num_p, matrix, node_data):
    curr_r = list(range(1, num_p + 1)); random.shuffle(curr_r)
    curr_d = calc_cost(curr_r, matrix, node_data)
    best_r, best_d, history = copy.deepcopy(curr_r), curr_d, []
    cooling, temp = 0.9176, 100.0

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

def optimize_GA(num_p, matrix, node_data):
    pop_size, mut_rate = 55, 0.32
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


# ==========================================
# METRICS EVALUATION EXECUTION
# ==========================================
komparasi_results = []
all_routes_map = {'SA': [], 'ACO': [], 'GA': [], 'LBA': []}
history_plot = {'SA': [], 'ACO': [], 'GA': [], 'LBA': []}

print("\n[PROSES] Memulai komparasi algoritma (VRPTW + SLA)...")

for i in range(len(df_grouped['trip_id'].unique())):
    df_segment = df_grouped[df_grouped['trip_id'] == i].copy()
    coords_trip = [DC_JUANDA] + df_segment['coords'].tolist()
    matrix = get_distance_matrix(coords_trip)
    num_p = len(df_segment)
    total_paket_trip = df_segment['jumlah_paket'].sum()
    total_pe_trip = df_segment[df_segment['is_priority'] == 1]['jumlah_paket'].count()

    print(f" -> Memproses Trip {i+1} ({num_p} titik, {total_paket_trip} paket, {total_pe_trip} prioritas PE)")

    node_data = {}
    for idx_nd, row_nd in df_segment.reset_index(drop=True).iterrows():
        node_data[idx_nd + 1] = {
            'service_time': row_nd['service_time'],
            'is_priority': row_nd['is_priority']
        }

    t0 = time.time(); best_r_sa, cost_sa, h_sa = optimize_SA(num_p, matrix, node_data); t_sa = time.time() - t0
    t0 = time.time(); best_r_aco, cost_aco, h_aco = optimize_ACO(num_p, matrix, node_data); t_aco = time.time() - t0
    t0 = time.time(); best_r_ga, cost_ga, h_ga = optimize_GA(num_p, matrix, node_data); t_ga = time.time() - t0
    t0 = time.time(); best_r_lba, cost_lba, h_lba = optimize_LBA(num_p, matrix, node_data); t_lba = time.time() - t0

    if i == 0:
        history_plot['SA'], history_plot['ACO'] = h_sa, h_aco
        history_plot['GA'], history_plot['LBA'] = h_ga, h_lba

    def process_route_data(algo_name, best_idx):
        r_coords = [DC_JUANDA] + [df_segment.iloc[idx-1]['coords'] for idx in best_idx] + [DC_JUANDA]
        full_geom = []
        current_time_min, total_dist_km, sla_violations = 0, 0, 0
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
    dist_aco, jam_aco, viol_aco = process_route_data('ACO', best_r_aco)
    dist_ga, jam_ga, viol_ga = process_route_data('GA', best_r_ga)
    dist_lba, jam_lba, viol_lba = process_route_data('LBA', best_r_lba)

    komparasi_results.append({
        'Trip': f'Trip {i+1}',
        'Dist_SA(km)': dist_sa, 'Dur_SA(jam)': jam_sa, 'Viol_SA': viol_sa, 'Time_SA(s)': t_sa,
        'Dist_ACO(km)': dist_aco, 'Dur_ACO(jam)': jam_aco, 'Viol_ACO': viol_aco, 'Time_ACO(s)': t_aco,
        'Dist_GA(km)': dist_ga, 'Dur_GA(jam)': jam_ga, 'Viol_GA': viol_ga, 'Time_GA(s)': t_ga,
        'Dist_LBA(km)': dist_lba, 'Dur_LBA(jam)': jam_lba, 'Viol_LBA': viol_lba, 'Time_LBA(s)': t_lba
    })

df_hasil = pd.DataFrame(komparasi_results)
df_hasil.loc['TOTAL'] = df_hasil.sum(numeric_only=True)
df_hasil.at['TOTAL', 'Trip'] = 'TOTAL KESELURUHAN'

print("\n[HASIL] Tabel Perbandingan Performa Komparasi Eksperimen:")
print(df_hasil.to_string())


# ==========================================
# VISUALIZATION INTERACTION GEOSPATIAL MAP
# ==========================================
map_komparasi = folium.Map(location=[DC_JUANDA[0], DC_JUANDA[1]], zoom_start=12, tiles='CartoDB positron')

folium.Marker(
    location=DC_JUANDA, 
    popup='<b>Depot DC Juanda</b>', 
    icon=folium.Icon(color='black', icon='home')
).add_to(map_komparasi)

# Palet warna segmen trip
trip_colors = {
    0: 'blue',       
    1: 'darkgreen',  
    2: 'purple',     
    3: 'cadetblue'   
}

for algo_name, trips_data in all_routes_map.items():
    is_default_show = True if algo_name == 'LBA' else False
    layer_group = folium.FeatureGroup(name=f"Rute & Marker ({algo_name})", show=is_default_show)
    
    for trip_idx, trip_info in enumerate(trips_data):
        geom_jalan = trip_info['geom']
        paket_markers = trip_info['paket']
        warna_jalur_trip = trip_colors.get(trip_idx, 'gray')
        
        if geom_jalan:
            folium.PolyLine(
                locations=geom_jalan, color=warna_jalur_trip, weight=5, opacity=0.8,
                tooltip=f"{algo_name} - Trip {trip_idx+1}"
            ).add_to(layer_group)
            
        for p in paket_markers:
            if p['is_pe'] == 1:
                warna_marker, icon_shape, status_prio = 'red', 'star', 'Prioritas Utama (PE)'
            else:
                warna_marker, icon_shape, status_prio = 'orange', 'info-sign', 'Reguler'
                
            waktu_est_wib = datetime.strptime("08:00", "%H:%M") + timedelta(minutes=p['waktu_sampai'])
            waktu_str = waktu_est_wib.strftime("%H:%M WIB")
            
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px; width: 220px;">
                <h4 style="margin:0 0 5px 0; color:{warna_jalur_trip};">{algo_name} - Trip {trip_idx+1}</h4>
                <hr style="margin: 3px 0; border: 0.5px solid #ccc;">
                <b>Urutan:</b> Ke-{p['urutan']}<br>
                <b>Estimasi Tiba:</b> <span style="color:green; font-weight:bold;">{waktu_str}</span><br>
                <b>SLA Kategori:</b> {status_prio}
            </div>
            """
            folium.Marker(
                location=p['koordinat'], icon=folium.Icon(color=warna_marker, icon=icon_shape),
                tooltip=f"[{algo_name} T{trip_idx+1}] Stop {p['urutan']} ({waktu_str})",
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(layer_group)
            
    layer_group.add_to(map_komparasi)

folium.LayerControl(collapsed=False).add_to(map_komparasi)

# --- EXPORT INTERACTIVE DASHBOARD MAP ---
file_html_peta = 'Dashboard_Peta_Komparasi_EAS.html'
map_komparasi.save(file_html_peta)

print(f"\n[OUTPUT] File dashboard geosponsal tersimpan pada: '{file_html_peta}'")