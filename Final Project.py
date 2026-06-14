import sys
import subprocess

for lib in ['optuna', 'polyline', 'folium', 'pandas', 'numpy', 'matplotlib']:
    try:
        __import__(lib)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

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
import optuna

# --- KONFIGURASI PROYEK ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
MAX_ITER_ALL = 150

N_RUNS = 10  

# DATA & LOKASI
file_path = 'Data Kurir Paket Yudi Hiparni.csv' 
DC_JUANDA = (-7.3817573, 112.7544115)
route_cache = {}

# CONSTRAINTS
SLA_LIMIT_PE = 240   # Jam 12:00 WIB
SHIFT_LIMIT = 540    # Jam 17:00 WIB
FIXED_SERVICE = 3    # Service time 3 menit

def get_distance_matrix(coords):
    if len(coords) <= 100:
        coord_str = ";".join([f"{c[1]},{c[0]}" for c in coords])
        url = f"http://router.project-osrm.org/table/v1/driving/{coord_str}?sources=all&destinations=all&annotations=distance,duration"
        try:
            r = requests.get(url, timeout=30).json()
            if r.get('code') == 'Ok': 
                return np.array(r['distances']) / 1000, np.array(r['durations'])
        except: pass
    n = len(coords)
    mat_dist = np.zeros((n, n)); mat_time = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = math.radians(coords[i][0]), math.radians(coords[i][1])
                lat2, lon2 = math.radians(coords[j][0]), math.radians(coords[j][1])
                a = math.sin((lat2 - lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1)/2)**2
                d_km = 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                mat_dist[i][j] = d_km
                mat_time[i][j] = d_km * 120 
    return mat_dist, mat_time

def get_osrm_route_cached(p1, p2):
    cache_key = (round(p1[0],5), round(p1[1],5), round(p2[0],5), round(p2[1],5))
    if cache_key in route_cache: return route_cache[cache_key]
    url = f"http://router.project-osrm.org/route/v1/driving/{p1[1]},{p1[0]};{p2[1]},{p2[0]}?overview=full&geometries=polyline"
    try:
        r = requests.get(url, timeout=10).json()
        if r['code'] == 'Ok':
            geom = polyline.decode(r['routes'][0]['geometry'])
            return geom, r['routes'][0]['distance']/1000, r['routes'][0]['duration']/3600
    except: pass
    return [p1, p2], 0, 0

def load_and_group_data(path):
    try: data = pd.read_csv(path, sep=';', engine='python', encoding='utf-8-sig')
    except: data = pd.read_csv(path, sep=',', engine='python', encoding='utf-8-sig')
    data.columns = [str(c).strip() for c in data.columns]
    data['lat'] = data['Koordinat'].apply(lambda x: float(str(x).split(',')[0]))
    data['lon'] = data['Koordinat'].apply(lambda x: float(str(x).split(',')[1]))
    data = data.dropna(subset=['lat', 'lon']).reset_index(drop=True).head(141)
    data['is_priority'] = data['Produk'].apply(lambda x: 1 if str(x).upper().strip() == 'PE' else 0)
    df_g = data.groupby(['lat', 'lon']).agg({'Nosi': list, 'is_priority': 'max'}).reset_index()
    df_g['service_time'] = FIXED_SERVICE
    df_g['jumlah_paket'] = df_g['Nosi'].apply(len)
    df_g['coords'] = list(zip(df_g['lat'], df_g['lon']))
    return df_g

def calc_cost(route, matrix_dist, matrix_time, node_data):
    total_dist, curr_time, penalty, curr = 0, 0, 0, 0
    for nxt in route:
        total_dist += matrix_dist[curr][nxt]
        curr_time += (matrix_time[curr][nxt]/60) + node_data[nxt]['service_time']
        if node_data[nxt]['is_priority'] == 1 and curr_time > SLA_LIMIT_PE:
            penalty += 2000 * (curr_time - SLA_LIMIT_PE)
        curr = nxt
    curr_time += (matrix_dist[curr][0] * 2) 
    if curr_time > SHIFT_LIMIT: penalty += 5000 * (curr_time - SHIFT_LIMIT)
    return total_dist + penalty

# --- ALGORITHMS ---
def optimize_SA(num_p, md, mt, nd, cooling=0.9176, temp=100.0):
    curr_r = list(range(1, num_p + 1)); random.shuffle(curr_r)
    curr_d = calc_cost(curr_r, md, mt, nd)
    best_r, best_d, hist = copy.deepcopy(curr_r), curr_d, []
    for _ in range(MAX_ITER_ALL):
        new_r = copy.deepcopy(curr_r); i, j = random.sample(range(num_p), 2)
        new_r[i], new_r[j] = new_r[j], new_r[i]
        new_d = calc_cost(new_r, md, mt, nd)
        if new_d < curr_d or random.random() < math.exp(max(-100, (curr_d - new_d) / temp)):
            curr_r, curr_d = new_r, new_d
            if curr_d < best_d: best_r, best_d = copy.deepcopy(curr_r), curr_d
        hist.append(best_d); temp *= cooling
    return best_r, best_d, hist

def optimize_ACO(num_p, md, mt, nd, alpha=0.67, beta=2.0, evap=0.2):
    ph = np.ones((num_p+1, num_p+1))
    best_r, best_d, hist = None, float('inf'), []
    for _ in range(MAX_ITER_ALL):
        for _ in range(20):
            un = list(range(1, num_p+1)); curr, route = 0, []
            while un:
                pr = [(ph[curr][n]**alpha) * ((1.0/(md[curr][n]+0.001))**beta) for n in un]
                nxt = np.random.choice(un, p=np.array(pr)/sum(pr))
                route.append(nxt); un.remove(nxt); curr = nxt
            c = calc_cost(route, md, mt, nd)
            if c < best_d: best_r, best_d = route, c
            cur_n = 0
            for n in route: ph[cur_n][n] += 1.0/c; cur_n = n
        ph *= (1 - evap); hist.append(best_d)
    return best_r, best_d, hist

def optimize_GA(num_p, md, mt, nd, pop_s=50, m_r=0.3):
    pop = [random.sample(range(1, num_p+1), num_p) for _ in range(pop_s)]
    best_r, best_d, hist = None, float('inf'), []
    for _ in range(MAX_ITER_ALL):
        pop = sorted(pop, key=lambda x: calc_cost(x, md, mt, nd))
        c_best = calc_cost(pop[0], md, mt, nd)
        if c_best < best_d: best_r, best_d = copy.deepcopy(pop[0]), c_best
        hist.append(best_d)
        nxt_gen = pop[:10]
        while len(nxt_gen) < pop_s:
            p1, p2 = random.sample(pop[:20], 2); pt = random.randint(1, num_p-1)
            ch = p1[:pt] + [g for g in p2 if g not in p1[:pt]]
            if random.random() < m_r:
                i, j = random.sample(range(num_p), 2); ch[i], ch[j] = ch[j], ch[i]
            nxt_gen.append(ch)
        pop = nxt_gen
    return best_r, best_d, hist

def optimize_LBA(num_p, md, mt, nd, pop_s=50, max_l=15):
    pop = [random.sample(range(1, num_p+1), num_p) for _ in range(pop_s)]
    g_best_r, g_best_d, hist = None, float('inf'), []
    for _ in range(MAX_ITER_ALL):
        fit = [calc_cost(ind, md, mt, nd) for ind in pop]
        idx = np.argmin(fit)
        if fit[idx] < g_best_d: g_best_r, g_best_d = copy.deepcopy(pop[idx]), fit[idx]
        else:
            ls_r, ls_d = copy.deepcopy(g_best_r), g_best_d
            for _ in range(max_l):
                t_r = copy.deepcopy(ls_r); i, j = random.sample(range(num_p), 2)
                t_r[i], t_r[j] = t_r[j], t_r[i]
                td = calc_cost(t_r, md, mt, nd)
                if td < ls_d: ls_r, ls_d = t_r, td
            if ls_d < g_best_d: g_best_r, g_best_d = copy.deepcopy(ls_r), ls_d
        hist.append(g_best_d)
        new_p = [copy.deepcopy(g_best_r)]
        while len(new_p) < pop_s:
            S = random.choice(pop[:20])
            if random.random() < 0.9:
                i, j = sorted(random.sample(range(num_p), 2)); cand = copy.deepcopy(S)
                cand[i], cand[j] = cand[j], cand[i]
            else: cand = random.sample(range(1, num_p+1), num_p)
            new_p.append(cand)
        pop = new_p
    return g_best_r, g_best_d, hist

# --- LOGIKA ENGINE PARALEL RUNNER (STATISTIK MULTI-RUN VALID) ---
def single_run(algo_func, num_p, md, mt, nd, seed_val, **kwargs):
    random.seed(seed_val)
    np.random.seed(seed_val)
    t0 = time.time()
    r, c, h = algo_func(num_p, md, mt, nd, **kwargs)
    return r, c, h, (time.time() - t0)

def execute_statistical_runs(algo_func, num_p, md, mt, nd, **kwargs):
    costs, times, routes, histories = [], [], [], []
    base_seed = random.randint(0, 10000)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(single_run, algo_func, num_p, md, mt, nd, base_seed + run, **kwargs) for run in range(N_RUNS)]
        for fut in concurrent.futures.as_completed(futures):
            r, c, h, t_el = fut.result()
            costs.append(c); times.append(t_el); routes.append(r); histories.append(h)
    best_idx = np.argmin(costs)
    return routes[best_idx], costs[best_idx], np.mean(costs), np.std(costs), np.mean(times), histories[best_idx]

if __name__ == '__main__':
    df_grouped = load_and_group_data(file_path)
    trip_ids, cur_trip, cur_load = [], 0, 0
    for idx, row in df_grouped.iterrows():
        if cur_load + row['jumlah_paket'] > 50: cur_trip += 1; cur_load = 0
        trip_ids.append(cur_trip); cur_load += row['jumlah_paket']
    df_grouped['trip_id'] = trip_ids

    print(f"[PREPROCESSING] Total: {df_grouped['jumlah_paket'].sum()} paket | {len(df_grouped)} titik lokasi unik.")
    for t in df_grouped['trip_id'].unique():
        n_paket = df_grouped[df_grouped['trip_id']==t]['jumlah_paket'].sum()
        n_titik = len(df_grouped[df_grouped['trip_id']==t])
        n_pe = df_grouped[(df_grouped['trip_id']==t) & (df_grouped['is_priority']==1)]['jumlah_paket'].count()
        print(f" - Trip {t+1}: {n_paket} paket | {n_titik} titik antar | {n_pe} titik prioritas PE.")

    print(f"\n[PROSES] Memulai iterasi {N_RUNS}x Paralel dengan Auto-Tuning Hyperparameter...")

    komparasi_results = []
    all_routes = {'SA':[], 'ACO':[], 'GA':[], 'LBA':[]}
    hists = {'SA':[], 'ACO':[], 'GA':[], 'LBA':[]}

    for i in range(len(df_grouped['trip_id'].unique())):
        seg = df_grouped[df_grouped['trip_id'] == i].copy().reset_index(drop=True)
        md, mt = get_distance_matrix([DC_JUANDA] + seg['coords'].tolist())
        num_p = len(seg)
        nd = {idx+1: {'service_time': r['service_time'], 'is_priority': r['is_priority']} for idx, r in seg.iterrows()}

        print(f"\n -> Memproses Trip {i+1} ({num_p} titik, {df_grouped[df_grouped['trip_id']==i]['jumlah_paket'].sum()} paket)")
        print("    [*] Melakukan Tuning Parameter Seluruh Algoritma...")
        
        study_sa = optuna.create_study(direction='minimize')
        study_sa.optimize(lambda t: optimize_SA(num_p, md, mt, nd, t.suggest_float('c',0.85,0.98), t.suggest_float('t',50,150))[1], n_trials=10)
        print(f"    [+] SA Tuned Parameter -> Cooling: {study_sa.best_params['c']:.2f} | Temp: {study_sa.best_params['t']:.1f}")
        
        study_aco = optuna.create_study(direction='minimize')
        study_aco.optimize(lambda t: optimize_ACO(num_p, md, mt, nd, t.suggest_float('a',0.5,1.2), t.suggest_float('b',1.5,2.5))[1], n_trials=10)
        print(f"    [+] ACO Tuned Parameter -> Alpha: {study_aco.best_params['a']:.2f} | Beta: {study_aco.best_params['b']:.2f}")

        study_ga = optuna.create_study(direction='minimize')
        study_ga.optimize(lambda t: optimize_GA(num_p, md, mt, nd, t.suggest_int('p',30,70), t.suggest_float('m',0.1,0.4))[1], n_trials=10)
        print(f"    [+] GA Tuned Parameter -> Pop Size: {study_ga.best_params['p']} | Mut Rate: {study_ga.best_params['m']:.2f}")

        study_lba = optuna.create_study(direction='minimize')
        study_lba.optimize(lambda t: optimize_LBA(num_p, md, mt, nd, t.suggest_int('p',40,80))[1], n_trials=10)
        print(f"    [+] LBA Tuned Parameter -> Pop Size: {study_lba.best_params['p']}")

        print(f"    [*] Mengeksekusi Algoritma ({N_RUNS}x Run)...")
        trip_metrics = {'Trip': f'Trip {i+1}'}

        for alg, func, p in [('SA', optimize_SA, study_sa.best_params), ('ACO', optimize_ACO, study_aco.best_params), 
                             ('GA', optimize_GA, study_ga.best_params), ('LBA', optimize_LBA, study_lba.best_params)]:
            
            if alg == 'SA': r_f, b_c, a_c, s_c, t_el, h_f = execute_statistical_runs(func, num_p, md, mt, nd, cooling=p['c'], temp=p['t'])
            elif alg == 'ACO': r_f, b_c, a_c, s_c, t_el, h_f = execute_statistical_runs(func, num_p, md, mt, nd, alpha=p['a'], beta=p['b'])
            elif alg == 'GA': r_f, b_c, a_c, s_c, t_el, h_f = execute_statistical_runs(func, num_p, md, mt, nd, pop_s=p['p'], m_r=p['m'])
            else: r_f, b_c, a_c, s_c, t_el, h_f = execute_statistical_runs(func, num_p, md, mt, nd, pop_s=p['p'])
            
            # Perbaikan Pemetaan Node Koordinat OSRM agar LBA Mengikuti Jalur Jalan Raya Surabaya
            r_c = [DC_JUANDA] + [seg.iloc[node_idx-1]['coords'] for node_idx in r_f] + [DC_JUANDA]
            geom, d_km, t_h, c_min, p_markers = [], 0, 0, 0, []
            for j in range(len(r_c)-1):
                g, dk, th = get_osrm_route_cached(r_c[j], r_c[j+1])
                geom.extend(g); c_min += (th*60)
                if j < len(r_f):
                    c_min += FIXED_SERVICE
                    p_markers.append({'u':j+1, 'c':r_c[j+1], 'w':c_min, 'p':nd[r_f[j]]['is_priority']})
                    
            all_routes[alg].append({'geom':geom, 'p':p_markers})
            if i == 0: hists[alg] = h_f
            
            trip_metrics[f'{alg}_Best'] = b_c
            trip_metrics[f'{alg}_Avg'] = a_c 
            trip_metrics[f'{alg}_Std'] = s_c
            trip_metrics[f'{alg}_Time(s)'] = t_el
            trip_metrics[f'{alg}_Viol'] = sum(1 for pm in p_markers if pm['p'] == 1 and pm['w'] > SLA_LIMIT_PE)

        komparasi_results.append(trip_metrics)

    # --- HASIL KOMPARASI OUTPUT TERMINAL ---
    df_hasil = pd.DataFrame(komparasi_results)
    print("\n[HASIL] Tabel Perbandingan Performa:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_hasil.to_string(index=False))

    df_hasil.to_csv('Hasil_Komparasi_Algoritma.csv', index=False)
    print("\n[INFO] Data hasil komparasi telah disimpan ke 'Hasil_Komparasi_Algoritma.csv'")

    # OUTPUT 1: BAR CHART (KOMPARASI PERFORMA BEST VS AVG POPULASI METODE)
    algos = ['SA', 'ACO', 'GA', 'LBA']
    avg_best_costs = [df_hasil[f'{alg}_Best'].mean() for alg in algos]
    avg_avg_costs = [df_hasil[f'{alg}_Avg'].mean() for alg in algos]
    avg_exec_times = [df_hasil[f'{alg}_Time(s)'].mean() for alg in algos]

    x = np.arange(len(algos))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 5))
    bar1 = ax1.bar(x - width/2, avg_best_costs, width, label='Rata-rata Best Cost', color='skyblue')
    bar2 = ax1.bar(x + width/2, avg_avg_costs, width, label='Rata-rata Average Cost', color='royalblue')
    ax1.set_ylabel('Cost Rute (Jarak + Penalti)', color='black', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    line = ax2.plot(x, avg_exec_times, color='red', marker='o', linewidth=2, label='Rata-rata Running Time (s)')
    ax2.set_ylabel('Running Time (Detik)', color='red', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='red')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    plt.title(f'Komparasi Performa Algoritma ({N_RUNS}x Run Paralel per Trip)', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('Grafik_Komparasi_Algoritma.png', dpi=300)
    print("[OUTPUT] Gambar Grafik Performa tersimpan pada: 'Grafik_Komparasi_Algoritma.png'")

    # OUTPUT 2: LINE CHART (GRAFIK KONVERGENSI)
    plt.figure(figsize=(10, 5))
    for alg, color in [('SA','#ee7272'), ('ACO','#3498db'), ('GA','#2ecc71'), ('LBA','#f1c40f')]:
        plt.plot(hists[alg], label=alg, color=color, linewidth=2)
    plt.yscale('log')
    plt.title('Grafik Konvergensi (Trip 1) - Skala Logaritmik')
    plt.xlabel('Iterasi'); plt.ylabel('Total Fitness Cost'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('Grafik_Konvergensi_Algoritma.png', dpi=300)
    print("[OUTPUT] Gambar Grafik Konvergensi tersimpan pada: 'Grafik_Konvergensi_Algoritma.png'")

    # --- MAP DASHBOARD ---
    m = folium.Map(location=[DC_JUANDA[0], DC_JUANDA[1]], zoom_start=12, tiles='CartoDB positron')
    tc = {0: "#ee7272", 1: "#f2dc60", 2: "#51e356"}
    
    folium.Marker(location=DC_JUANDA, popup='<b>Depot DC Juanda</b>', icon=folium.Icon(color='blue', icon='home')).add_to(m)
    
    for alg, trips in all_routes.items():
        lg = folium.FeatureGroup(name=f"{alg}", show=(alg=='LBA'))
        for idx, t in enumerate(trips):
            folium.PolyLine(t['geom'], color=tc.get(idx,'#2980b9'), weight=5, opacity=0.85, tooltip=f"{alg} - Trip {idx+1}", className=f"route-line line-algo-{alg} line-trip-{idx}").add_to(lg)
            for p in t['p']:
                icon_c = '#e74c3c' if p['p'] else '#f39c12'
                w_str = (datetime.strptime("08:00","%H:%M")+timedelta(minutes=p['w'])).strftime("%H:%M")
                
                folium.Marker(p['c'], icon=folium.DivIcon(html=f'<div class="custom-marker algo-{alg} trip-{idx} {"type-pe" if p["p"] else "type-reg"}" style="background:{icon_c};color:white;border-radius:50%;width:22px;height:22px;display:flex;justify-content:center;align-items:center;font-weight:bold;font-size:10px;border:2px solid white;box-shadow:0px 4px 10px rgba(0,0,0,0.25);">{p["u"]}</div>'),
                popup=f"<b>{alg} Trip {idx+1}</b><br>Urutan: {p['u']}<br>Estimasi Tiba: {w_str} WIB").add_to(lg)
        lg.add_to(m)
    
    folium.LayerControl(position='topright', collapsed=False).add_to(m)

    header_html = f'''
    <div style="position:fixed; top:0; left:0; width:100%; height:65px; background:linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); color:#f1f5f9; z-index:9999; display:flex; align-items:center; padding-left:25px; font-family:'Segoe UI', Roboto, Arial; box-shadow:0 4px 20px rgba(0,0,0,0.3); border-bottom: 2px solid #00d2ff;">
        <h2 style="margin:0; font-size:20px; font-weight:600; letter-spacing:0.8px;">DASHBOARD RUTE KURIR PENGANTARAN PAKET</h2>
    </div>
    <style>
        .leaflet-top.leaflet-right {{ 
            margin-top: 85px !important; 
            margin-right: 30px !important;
        }}
        .leaflet-control-layers {{
            background: rgba(15, 23, 42, 0.88) !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            border: 1px solid rgba(0, 210, 255, 0.3) !important;
            border-radius: 16px !important;
            box-shadow: 0 12px 36px rgba(0,0,0,0.3) !important;
            color: #f1f5f9 !important;
            font-family: 'Segoe UI', Roboto, Arial !important;
            padding: 20px !important;
            font-size: 13px !important;
            width: 255px !important;
            box-sizing: border-box !important;
            animation: fadeIn 0.7s cubic-bezier(0.25, 1, 0.5, 1) forwards;
        }}
        .leaflet-control-layers-base {{
            display: none !important; 
        }}
        .leaflet-control-layers-overlays::before {{
            content: "METODE" !important;
            display: block !important;
            font-weight: bold !important;
            color: #00d2ff !important;
            font-size: 13px !important;
            margin-bottom: 8px !important;
            letter-spacing: 0.5px !important;
            border-bottom: 1px solid rgba(0, 210, 255, 0.25) !important;
            padding-bottom: 4px !important;
        }}
        .leaflet-control-layers-overlays label {{
            margin-bottom: 8px !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            color: #ffffff !important;
        }}
        .leaflet-control-layers input {{
            margin-right: 8px !important;
            cursor: pointer !important;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .modern-dashboard-panel {{
            animation: fadeIn 0.7s cubic-bezier(0.25, 1, 0.5, 1) forwards;
        }}
    </style>
    '''
    
    filter_html = '''
    <div class="modern-dashboard-panel" style="position:fixed; bottom:35px; right:30px; width:255px; background:rgba(15, 23, 42, 0.88); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); border-radius:16px; padding:20px; z-index:9999; font-family:'Segoe UI', Roboto, Arial; box-shadow:0 12px 36px rgba(0,0,0,0.3); border:1px solid rgba(0, 210, 255, 0.3); color:#f1f5f9; box-sizing: border-box;">
        <b style="color:#00d2ff; font-size:13px; display:block; margin-bottom:8px; letter-spacing:0.5px;">TRIP</b>
        <hr style="border:0; border-top:1px solid rgba(0, 210, 255, 0.25); margin:6px 0 12px 0;">
        <div style="margin-bottom:8px;"><input type="checkbox" id="t0" checked onchange="up()" style="cursor:pointer;"> <label for="t0" style="color:#ffffff;font-weight:600;cursor:pointer;margin-left:6px;">Trip 1</label></div>
        <div style="margin-bottom:8px;"><input type="checkbox" id="t1" checked onchange="up()" style="cursor:pointer;"> <label for="t1" style="color:#ffffff;font-weight:600;cursor:pointer;margin-left:6px;">Trip 2</label></div>
        <div style="margin-bottom:14px;"><input type="checkbox" id="t2" checked onchange="up()" style="cursor:pointer;"> <label for="t2" style="color:#ffffff;font-weight:600;cursor:pointer;margin-left:6px;">Trip 3</label></div>
        <b style="color:#00d2ff; font-size:13px; display:block; margin-bottom:8px; letter-spacing:0.5px;">KLASIFIKASI PAKET</b>
        <hr style="border:0; border-top:1px solid rgba(0, 210, 255, 0.25); margin:6px 0 12px 0;">
        <div style="margin-bottom:8px;"><input type="checkbox" id="reg" checked onchange="up()" style="cursor:pointer;"> <label for="reg" style="cursor:pointer;margin-left:6px;">Paket Reguler <span style="color:#f39c12">●</span></label></div>
        <div><input type="checkbox" id="pe" checked onchange="up()" style="cursor:pointer;"> <label for="pe" style="cursor:pointer;margin-left:6px;">Paket Ekspres <span style="color:#e74c3c">●</span></label></div>
    </div>
    <script>
    function up(){
        var s0=document.getElementById("t0").checked, s1=document.getElementById("t1").checked, s2=document.getElementById("t2").checked;
        var rg=document.getElementById("reg").checked, pe=document.getElementById("pe").checked;
        
        document.querySelectorAll(".custom-marker").forEach(m => {
            var t = (m.classList.contains("trip-0")&&s0)||(m.classList.contains("trip-1")&&s1)||(m.classList.contains("trip-2")&&s2);
            var ty = (m.classList.contains("type-reg")&&rg)||(m.classList.contains("type-pe")&&pe);
            if(m.parentNode) m.parentNode.style.display = (t && ty) ? "block" : "none";
        });
        
        document.querySelectorAll(".route-line").forEach(line => {
            var is_t0 = line.classList.contains("line-trip-0");
            var is_t1 = line.classList.contains("line-trip-1");
            var is_t2 = line.classList.contains("line-trip-2");
            var showLine = (is_t0 && s0) || (is_t1 && s1) || (is_t2 && s2);
            line.style.display = showLine ? "block" : "none";
        });
    }
    var obs = new MutationObserver(up);
    obs.observe(document.body, { childList: true, subtree: true });
    setTimeout(up, 600);
    </script>'''
    
    m.get_root().html.add_child(folium.Element(header_html + filter_html))
    m.save('Dashboard_Rute_EAS_Final.html')
    print("[OUTPUT] File dashboard geospatial tersimpan pada: 'Dashboard_Rute_EAS_Final.html'")