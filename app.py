import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import io

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Smart Pile & Foundation Design Suite",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ AI-Assisted Boring Log to Pile Design & Load Transfer Simulator")
st.markdown("ระบบออกแบบเสาเข็มและฐานรากครบวงจร: จำลองการถ่ายแรง ($t-z / q-z$), วิเคราะห์ความเสี่ยง (Monte Carlo), ออกแบบ Pile Cap (SDM) และวินิจฉัยชั้นดินด้วย AI")

# ==========================================
# 2. GEOTECHNICAL & LOAD TRANSFER ENGINE
# ==========================================
def get_soil_parameters(layer):
    soil_type = str(layer['Soil Type']).strip()
    N = float(layer['SPT_N'])
    gamma = float(layer['Unit Weight (kN/m3)'])
    
    if soil_type.lower() == 'clay':
        Su = max(10.0, 6.0 * N)
        alpha = 1.0 if Su <= 25 else (0.5 if Su >= 75 else 1.0 - 0.5 * ((Su - 25) / 50))
        fs_ult = alpha * Su
        qb_ult = 9.0 * Su
        return {'Su': Su, 'phi': 0.0, 'fs_ult': fs_ult, 'qb_ult': qb_ult, 'gamma': gamma}
    else:
        phi = min(45.0, 27.1 + 0.3 * N - 0.00054 * (N ** 2))
        phi_rad = np.radians(phi)
        Ks = 0.7
        delta = 0.75 * phi_rad
        Nq = np.exp(np.pi * np.tan(phi_rad)) * (np.tan(np.radians(45 + phi / 2)) ** 2)
        return {'Su': 0.0, 'phi': phi, 'Ks': Ks, 'delta': delta, 'Nq': Nq, 'gamma': gamma}

def tz_curve_api(z, z_peak, t_max, soil_type):
    if z <= 0:
        return 0.0
    ratio = z / z_peak
    if soil_type.lower() == 'clay':
        if ratio <= 0.16:
            return 0.30 * t_max * (ratio / 0.16)
        elif ratio <= 0.31:
            return (0.30 + 0.20 * ((ratio - 0.16) / 0.15)) * t_max
        elif ratio <= 0.57:
            return (0.50 + 0.25 * ((ratio - 0.31) / 0.26)) * t_max
        elif ratio <= 0.80:
            return (0.75 + 0.15 * ((ratio - 0.57) / 0.23)) * t_max
        elif ratio <= 1.00:
            return (0.90 + 0.10 * ((ratio - 0.80) / 0.20)) * t_max
        elif ratio <= 2.00:
            return (1.00 - 0.20 * ((ratio - 1.00) / 1.00)) * t_max
        else:
            return 0.80 * t_max
    else:
        if ratio < 1.0:
            return t_max * np.sin((np.pi / 2.0) * ratio)
        else:
            return t_max

def qz_curve_api(z_tip, D, q_max):
    if z_tip <= 0:
        return 0.0
    z_ult = 0.10 * D
    ratio = z_tip / z_ult
    if ratio <= 0.02:
        return 0.25 * q_max * (ratio / 0.02)
    elif ratio <= 0.13:
        return (0.25 + 0.25 * ((ratio - 0.02) / 0.11)) * q_max
    elif ratio <= 0.42:
        return (0.50 + 0.25 * ((ratio - 0.13) / 0.29)) * q_max
    elif ratio <= 1.00:
        return (0.75 + 0.25 * ((ratio - 0.42) / 0.58)) * q_max
    else:
        return q_max

def run_load_transfer_simulation(df_soil, D, L, Ep_GPa, gwt_depth, n_elements=40):
    dz = L / n_elements
    area = np.pi * (D ** 2) / 4.0
    perimeter = np.pi * D
    Ep = Ep_GPa * 1e6
    
    elem_depths = np.linspace(dz / 2.0, L - dz / 2.0, n_elements)
    elem_soil_types = []
    elem_tmax = []
    sigma_v_eff = 0.0
    
    for d_mid in elem_depths:
        layer_match = df_soil[(df_soil['Top (m)'] <= d_mid) & (df_soil['Bottom (m)'] >= d_mid)]
        layer = df_soil.iloc[-1] if layer_match.empty else layer_match.iloc[0]
            
        params = get_soil_parameters(layer)
        gamma_eff = params['gamma'] - 9.81 if d_mid > gwt_depth else params['gamma']
        sigma_v_eff += gamma_eff * dz
        
        if str(layer['Soil Type']).strip().lower() == 'clay':
            fs = params['fs_ult']
        else:
            fs = min(120.0, params['Ks'] * sigma_v_eff * np.tan(params['delta']))
            
        elem_soil_types.append(str(layer['Soil Type']).strip())
        elem_tmax.append(fs * perimeter * dz)
        
    tip_layer_match = df_soil[(df_soil['Top (m)'] <= L) & (df_soil['Bottom (m)'] >= L)]
    tip_layer = df_soil.iloc[-1] if tip_layer_match.empty else tip_layer_match.iloc[0]
    tip_params = get_soil_parameters(tip_layer)
    
    if str(tip_layer['Soil Type']).strip().lower() == 'clay':
        qb_ult = tip_params['qb_ult']
    else:
        qb_ult = min(10000.0, tip_params['Nq'] * sigma_v_eff)
    Qb_max = qb_ult * area
    Qs_max = sum(elem_tmax)
    Q_ult_total = Qs_max + Qb_max

    z_tips = np.linspace(0.0001, 0.12 * D, 50)
    top_loads = []
    top_settlements = []
    profiles = []
    z_peak = 0.01 * D
    
    for z_t in z_tips:
        Q_bot = qz_curve_api(z_t, D, Qb_max)
        z_curr = z_t
        depth_nodes = [L]
        load_nodes = [Q_bot]
        
        for i in range(n_elements - 1, -1, -1):
            z_mid = z_curr
            for _ in range(5):
                T_side = tz_curve_api(z_mid, z_peak, elem_tmax[i], elem_soil_types[i])
                Q_top = Q_bot + T_side
                Q_avg = 0.5 * (Q_bot + Q_top)
                delta_z = (Q_avg * dz) / (area * Ep)
                z_mid = z_curr + 0.5 * delta_z
                
            z_curr += delta_z
            Q_bot = Q_top
            depth_nodes.append(elem_depths[i] - dz / 2.0)
            load_nodes.append(Q_top)
            
        top_loads.append(Q_bot)
        top_settlements.append(z_curr * 1000.0)
        profiles.append((depth_nodes[::-1], load_nodes[::-1]))
        
    return {
        'Q_ult': Q_ult_total,
        'Qs_max': Qs_max,
        'Qb_max': Qb_max,
        'top_loads': np.array(top_loads),
        'top_settlements': np.array(top_settlements),
        'profiles': profiles,
        'tip_layer': tip_layer
    }

def run_monte_carlo_reliability(Qs_mean, Qb_mean, P_design, cov_soil=0.20, cov_load=0.10, n_sim=2000):
    np.random.seed(42)
    R_mean = Qs_mean + Qb_mean
    R_std = R_mean * cov_soil
    sigma_ln_R = np.sqrt(np.log(1 + (R_std / R_mean) ** 2))
    mu_ln_R = np.log(R_mean) - 0.5 * (sigma_ln_R ** 2)
    R_samples = np.random.lognormal(mu_ln_R, sigma_ln_R, n_sim)
    
    Q_std = P_design * cov_load
    Q_samples = np.random.normal(P_design, Q_std, n_sim)
    
    g = R_samples - Q_samples
    failures = np.sum(g <= 0)
    Pf = (failures / n_sim) * 100.0
    beta = (np.mean(g)) / np.std(g)
    return R_samples, Q_samples, Pf, beta

# ==========================================
# 3. AI DIAGNOSTIC & OPTIMIZER ENGINE
# ==========================================
def ai_soil_diagnostic(df_soil, D, L, P_design, FS_req, sim_res, settlement_at_P, Pf):
    insights = []
    warnings = []
    
    # 1. ตรวจจับชั้นดินอ่อน (Soft Clay N <= 4)
    soft_layers = df_soil[(df_soil['Soil Type'].str.lower() == 'clay') & (df_soil['SPT_N'] <= 4)]
    if not soft_layers.empty:
        max_soft_depth = soft_layers['Bottom (m)'].max()
        warnings.append(f"⚠️ **พบชั้นดินเหนียวอ่อนมาก (Soft Clay, N ≤ 4) ลึกถึง {max_soft_depth:.1f} ม.:** เสี่ยงต่อการเกิดแรงฉุดลง (Negative Skin Friction) หากมีการถมดินเพิ่ม และมีกำลังต้านทานด้านข้างต่ำมาก")
        if L <= max_soft_depth + 2.0:
            warnings.append(f"🚨 **ความยาวเสาเข็ม ({L:.1f} ม.) สั้นเกินไป:** ปลายเสาเข็มยังอยู่ในหรือใกล้ชั้นดินอ่อนมาก ควรเพิ่มความยาวให้หยั่งลึกถึงชั้นทรายแน่นด้านล่าง")
            
    # 2. ตรวจสอบชั้นดินรองรับปลายเสาเข็ม (Bearing Stratum)
    tip_N = float(sim_res['tip_layer']['SPT_N'])
    tip_type = sim_res['tip_layer']['Soil Type']
    if tip_N < 15:
        warnings.append(f"⚠️ **ชั้นดินปลายเสาเข็ม ({tip_type}, SPT-N = {tip_N:.0f}) ยังไม่แน่นพอ:** แนะนำให้เพิ่มความยาวเสาเข็มจนถึงชั้นดินแข็งที่มีค่า N ≥ 20–30 เพื่อลดการทรุดตัวระยะยาว")
    else:
        insights.append(f"✅ **ชั้นดินรองรับปลายเสาเข็มเหมาะสม:** ปลายเสาเข็มวางตัวในชั้น {tip_type} (SPT-N = {tip_N:.0f}) ให้แรงต้านทานปลายเสาเข็มสูง ({sim_res['Qb_max']:,.0f} kN)")

    # 3. ตรวจสอบสัดส่วน Skin vs End Bearing
    skin_ratio = (sim_res['Qs_max'] / sim_res['Q_ult']) * 100
    insights.append(f"ℹ️ **พฤติกรรมการรับน้ำหนัก:** เสาเข็มรับแรงด้วยแรงเสียดทานด้านข้าง (Skin Friction) **{skin_ratio:.1f}%** และแรงต้านทานที่ปลาย (End Bearing) **{100-skin_ratio:.1f}%**")

    # 4. ตรวจสอบความปลอดภัยและระยะทรุดตัว
    FS_actual = sim_res['Q_ult'] / P_design
    if FS_actual < FS_req:
        warnings.append(f"🚨 **กำลังรับน้ำหนักไม่เพียงพอ (FS = {FS_actual:.2f} < {FS_req:.1f}):** ควรเพิ่มเส้นผ่านศูนย์กลาง (D) หรือเพิ่มความยาวเสาเข็ม (L)")
    if settlement_at_P > 25.0:
        warnings.append(f"🚨 **การทรุดตัวเกินเกณฑ์มาตรฐาน ({settlement_at_P:.1f} mm > 25 mm):** โครงสร้างส่วนบนอาจแตกร้าวจากการทรุดตัวต่างระดับ")
    if Pf > 1.0:
        warnings.append(f"⚠️ **ความเสี่ยงเชิงความน่าจะเป็นสูง (Pf = {Pf:.2f}%):** แม้ค่า FS อาจดูใกล้เคียงเกณฑ์ แต่ความแปรปรวนของชั้นดินทำให้มีโอกาสวิบัติเกิน 1%")

    # 5. AI Auto-Optimizer: ค้นหาขนาด D และ L ที่เหมาะสมที่สุด
    best_design = None
    candidate_Ds = [0.35, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20]
    max_soil_depth = float(df_soil['Bottom (m)'].max())
    candidate_Ls = np.arange(8.0, min(40.0, max_soil_depth + 2.0), 1.0)
    
    for d_try in candidate_Ds:
        for l_try in candidate_Ls:
            res_try = run_load_transfer_simulation(df_soil, d_try, l_try, 25.0, 2.5, n_elements=25)
            if res_try['Q_ult'] / P_design >= FS_req:
                settle_try = np.interp(P_design, res_try['top_loads'], res_try['top_settlements'])
                if settle_try <= 20.0 and float(res_try['tip_layer']['SPT_N']) >= 15:
                    vol = (np.pi * (d_try**2) / 4.0) * l_try
                    if best_design is None or vol < best_design['vol']:
                        best_design = {'D': d_try, 'L': l_try, 'Q_ult': res_try['Q_ult'], 'settle': settle_try, 'vol': vol}
                    break
                    
    return insights, warnings, best_design

# ==========================================
# 4. PILE GROUP & PILE CAP DESIGNER (SDM)
# ==========================================
def design_pile_cap_sdm(P_col_service, Mx_service, My_service, Q_allow_single, D, fc_ksc=240, fy_ksc=4000, col_b=0.50, col_h=0.50):
    """ออกแบบกลุ่มเสาเข็มและฐานรากหัวเสาเข็มตามวิธีหน่วยแรงประลัย (SDM - ว.ส.ท.)"""
    # 1. หาจำนวนเสาเข็มที่ต้องการ
    n_min = int(np.ceil((P_col_service * 1.10) / Q_allow_single))  # เผื่อน้ำหนักฐานราก 10%
    valid_configs = [1, 2, 3, 4, 5, 6, 8, 9]
    n_piles = next((n for n in valid_configs if n >= max(1, n_min)), 9)
    
    spacing = 3.0 * D
    edge_dist = max(0.35, 1.25 * D)
    
    # กำหนดพิกัดเสาเข็ม (x, y) และขนาดฐานราก (B x L_cap)
    coords = []
    if n_piles == 1:
        coords = [(0.0, 0.0)]
        B_cap = L_cap = max(col_b + 0.5, 2 * edge_dist)
    elif n_piles == 2:
        coords = [(-spacing/2, 0.0), (spacing/2, 0.0)]
        L_cap = spacing + 2 * edge_dist
        B_cap = max(col_b + 0.5, 2 * edge_dist)
    elif n_piles == 3:
        r = spacing / np.sqrt(3)
        coords = [(0.0, r), (-spacing/2, -r/2), (spacing/2, -r/2)]
        B_cap = spacing + 2 * edge_dist
        L_cap = (r + r/2) + 2 * edge_dist
    elif n_piles == 4:
        coords = [(-spacing/2, spacing/2), (spacing/2, spacing/2), (-spacing/2, -spacing/2), (spacing/2, -spacing/2)]
        B_cap = L_cap = spacing + 2 * edge_dist
    elif n_piles == 5:
        s_diag = spacing * 0.85
        coords = [(0.0, 0.0), (-s_diag, s_diag), (s_diag, s_diag), (-s_diag, -s_diag), (s_diag, -s_diag)]
        B_cap = L_cap = 2 * s_diag + 2 * edge_dist
    elif n_piles == 6:
        coords = [(-spacing, spacing/2), (0.0, spacing/2), (spacing, spacing/2),
                  (-spacing, -spacing/2), (0.0, -spacing/2), (spacing, -spacing/2)]
        L_cap = 2 * spacing + 2 * edge_dist
        B_cap = spacing + 2 * edge_dist
    elif n_piles == 8:
        coords = [(-1.5*spacing, spacing/2), (-0.5*spacing, spacing/2), (0.5*spacing, spacing/2), (1.5*spacing, spacing/2),
                  (-1.5*spacing, -spacing/2), (-0.5*spacing, -spacing/2), (0.5*spacing, -spacing/2), (1.5*spacing, -spacing/2)]
        L_cap = 3 * spacing + 2 * edge_dist
        B_cap = spacing + 2 * edge_dist
    else:  # 9 piles
        for ix in [-1, 0, 1]:
            for iy in [-1, 0, 1]:
                coords.append((ix * spacing, iy * spacing))
        B_cap = L_cap = 2 * spacing + 2 * edge_dist

    # 2. คำนวณประสิทธิภาพกลุ่มเสาเข็ม (Converse-Labarre Formula)
    if n_piles == 1:
        eta_g = 1.0
    else:
        theta_deg = np.degrees(np.arctan(D / spacing))
        m_rows = 2 if n_piles in [2, 4, 6, 8] else 3
        n_cols = int(np.ceil(n_piles / m_rows))
        eta_g = 1.0 - (theta_deg / 90.0) * (((n_cols - 1) * m_rows + (m_rows - 1) * n_cols) / (m_rows * n_cols))
        eta_g = max(0.65, min(1.0, eta_g))
        
    Q_allow_pile_eff = Q_allow_single * eta_g

    # 3. คำนวณแรงลงเสาเข็มแต่ละต้น (รวมผลของโมเมนต์ Mx, My)
    sum_x2 = sum(x**2 for x, y in coords)
    sum_y2 = sum(y**2 for x, y in coords)
    pile_loads = []
    for x, y in coords:
        P_i = (P_col_service / n_piles)
        if sum_x2 > 0:
            P_i += (My_service * x) / sum_x2
        if sum_y2 > 0:
            P_i += (Mx_service * y) / sum_y2
        pile_loads.append(P_i)
    max_pile_load = max(pile_loads)

    # 4. ออกแบบความหนาฐานรากและเหล็กเสริม (SDM)
    Pu_max = 1.45 * max_pile_load  # Ultimate Pile Reaction (kN)
    fc_mpa = fc_ksc * 0.0981
    fy_mpa = fy_ksc * 0.0981
    
    # ประมาณความหนาประสิทธิผล d จาก Punching Shear around Column
    b0_est = 2 * (col_b + col_h + 1.2)
    vc_punch = 0.33 * np.sqrt(fc_mpa) * 1000  # kPa (phi=0.85 included approx)
    d_req = max(0.45, (1.45 * P_col_service) / (0.85 * vc_punch * b0_est))
    d_eff = np.ceil(d_req * 20) / 20.0  # ปัดขึ้นทีละ 5 cm
    H_cap = d_eff + 0.10  # รวมระยะหุ้มคอนกรีต 10 cm
    
    # คำนวณโมเมนต์ดัดที่ขอบเสาตอม่อและเหล็กเสริม As
    arm = max(0.1, (L_cap / 2.0) - (col_h / 2.0) - edge_dist)
    Mu = Pu_max * (n_piles / 2.0) * arm if n_piles > 1 else Pu_max * 0.15
    As_req_cm2 = max(12.0, (Mu * 1e6) / (0.90 * fy_mpa * 0.90 * d_eff * 1000) / 100.0)
    
    # จัดเหล็กเสริม DB20 (พื้นที่หน้าตัด 3.14 cm2/เส้น)
    n_bars = int(np.ceil(As_req_cm2 / 3.14))
    bar_spacing_cm = ((B_cap - 0.15) * 100) / max(1, n_bars - 1)
    
    return {
        'n_piles': n_piles,
        'coords': coords,
        'B_cap': B_cap,
        'L_cap': L_cap,
        'H_cap': H_cap,
        'd_eff': d_eff,
        'spacing': spacing,
        'eta_g': eta_g,
        'Q_allow_pile_eff': Q_allow_pile_eff,
        'pile_loads': pile_loads,
        'max_pile_load': max_pile_load,
        'As_req_cm2': As_req_cm2,
        'n_bars': n_bars,
        'bar_spacing_cm': bar_spacing_cm
    }

# ==========================================
# 5. HTML / PRINT-TO-PDF REPORT GENERATOR
# ==========================================
def generate_html_report(project_name, D, L, Ep_GPa, gwt_depth, P_design, FS_req, df_soil, sim_res, settlement_at_P, Pf, beta, cap_res):
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    soil_rows = ""
    for _, r in df_soil.iterrows():
        soil_rows += f"<tr><td>{r['Top (m)']:.1f} - {r['Bottom (m)']:.1f}</td><td>{r['Soil Type']}</td><td>{r['SPT_N']}</td><td>{r['Unit Weight (kN/m3)']:.1f}</td></tr>"
        
    html = f"""
    <!DOCTYPE html>
    <html lang="th">
    <head>
    <meta charset="UTF-8">
    <title>รายการคำนวณออกแบบเสาเข็มและฐานราก - {project_name}</title>
    <style>
        body {{ font-family: 'Sarabun', 'Segoe UI', sans-serif; color: #222; line-height: 1.6; max-width: 900px; margin: 20px auto; padding: 20px; }}
        h1, h2, h3 {{ color: #1a365d; border-bottom: 2px solid #e2e8f0; padding-bottom: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #cbd5e0; padding: 8px 12px; text-align: center; }}
        th {{ background-color: #edf2f7; font-weight: bold; }}
        .summary-box {{ background: #f7fafc; border-left: 5px solid #2b6cb0; padding: 15px; margin: 15px 0; }}
        .pass {{ color: #2f855a; font-weight: bold; }}
        .fail {{ color: #c53030; font-weight: bold; }}
    </style>
    </head>
    <body>
        <h1>รายงานรายการคำนวณออกแบบเสาเข็มและฐานราก (Geotechnical & Foundation Report)</h1>
        <p><strong>ชื่อโครงการ:</strong> {project_name} | <strong>วันที่ออกรายงาน:</strong> {now_str}</p>
        
        <h2>1. ข้อมูลพารามิเตอร์เสาเข็มและน้ำหนักบรรทุก</h2>
        <table>
            <tr><th>เส้นผ่านศูนย์กลาง (D)</th><th>ความยาวเสาเข็ม (L)</th><th>โมดูลัสคอนกรีต (Ep)</th><th>ระดับน้ำใต้ดิน (GWT)</th><th>น้ำหนักบรรทุกต่อต้น (P)</th><th>FS ที่ต้องการ</th></tr>
            <tr><td>{D:.2f} m</td><td>{L:.2f} m</td><td>{Ep_GPa:.1f} GPa</td><td>{gwt_depth:.2f} m</td><td>{P_design:,.1f} kN</td><td>{FS_req:.2f}</td></tr>
        </table>

        <h2>2. ข้อมูลชั้นดินจากหลุมเจาะ (Boring Log Summary)</h2>
        <table>
            <tr><th>ช่วงความลึก (m)</th><th>ชนิดดิน (Soil Type)</th><th>ค่า SPT-N (blows/ft)</th><th>หน่วยน้ำหนักดิน (kN/m³)</th></tr>
            {soil_rows}
        </table>

        <h2>3. ผลการวิเคราะห์กำลังรับน้ำหนักและพฤติกรรมการถ่ายแรง (t-z / q-z Analysis)</h2>
        <div class="summary-box">
            <ul>
                <li><strong>แรงเสียดทานด้านข้างประลัย (Ultimate Skin Friction, Qs):</strong> {sim_res['Qs_max']:,.1f} kN</li>
                <li><strong>แรงต้านทานปลายเสาเข็มประลัย (Ultimate End Bearing, Qb):</strong> {sim_res['Qb_max']:,.1f} kN</li>
                <li><strong>กำลังรับน้ำหนักประลัยรวม (Total Ultimate Capacity, Q_ult):</strong> {sim_res['Q_ult']:,.1f} kN</li>
                <li><strong>กำลังรับน้ำหนักปลอดภัย (Allowable Capacity, Q_allow):</strong> {sim_res['Q_ult']/FS_req:,.1f} kN (FS จริง = {sim_res['Q_ult']/P_design:.2f})</li>
                <li><strong>การทรุดตัวที่หัวเสาเข็ม ณ น้ำหนักใช้งาน:</strong> {settlement_at_P:.2f} mm</li>
                <li><strong>การวิเคราะห์ความเสี่ยงเชิงความน่าจะเป็น (Monte Carlo):</strong> โอกาสวิบัติ Pf = {Pf:.2f}% | Reliability Index β = {beta:.2f}</li>
            </ul>
        </div>

        <h2>4. สรุปผลการออกแบบกลุ่มเสาเข็มและฐานราก (Pile Cap Design - SDM)</h2>
        <table>
            <tr><th>จำนวนเสาเข็ม</th><th>ประสิทธิภาพกลุ่ม (ηg)</th><th>ขนาดฐานราก (B × L × H)</th><th>แรงสูงสุดต่อต้น</th><th>เหล็กเสริมหลัก (แต่ละทิศทาง)</th></tr>
            <tr>
                <td>{cap_res['n_piles']} ต้น (@ {cap_res['spacing']:.2f} m)</td>
                <td>{cap_res['eta_g']:.3f}</td>
                <td>{cap_res['B_cap']:.2f} × {cap_res['L_cap']:.2f} × {cap_res['H_cap']:.2f} m</td>
                <td>{cap_res['max_pile_load']:,.1f} kN (Allow: {cap_res['Q_allow_pile_eff']:,.1f} kN)</td>
                <td>{cap_res['n_bars']}-DB20 @ {cap_res['bar_spacing_cm']:.1f} cm</td>
            </tr>
        </table>
        <p style="font-size: 0.9em; color: #666; margin-top: 30px;">*หมายเหตุ: สามารถกด Ctrl+P (หรือ Command+P) ในเบราว์เซอร์เพื่อบันทึกหน้านี้เป็นไฟล์ PDF ฉบับสมบูรณ์ได้ทันที</p>
    </body>
    </html>
    """
    return html

# ==========================================
# 6. SIDEBAR INPUTS (USER INTERFACE)
# ==========================================
with st.sidebar:
    st.header("1. ข้อมูลโครงการและเสาเข็ม")
    project_name = st.text_input("ชื่อโครงการ / อาคาร", value="อาคารเรียนรวม มทส. (SUT Project)")
    D = st.number_input("เส้นผ่านศูนย์กลางเสาเข็ม D (m)", min_value=0.20, max_value=2.00, value=0.60, step=0.05)
    L = st.number_input("ความยาวเสาเข็ม L (m)", min_value=3.0, max_value=50.0, value=16.0, step=0.5)
    Ep_GPa = st.number_input("โมดูลัสยืดหยุ่นคอนกรีต Ep (GPa)", min_value=15.0, max_value=45.0, value=25.0, step=1.0)
    gwt_depth = st.number_input("ระดับน้ำใต้ดิน GWT (m)", min_value=0.0, max_value=30.0, value=2.5, step=0.5)
    
    st.divider()
    st.header("2. แรงกระทำเสาเข็มเดี่ยว & ความแปรปรวน")
    P_design = st.number_input("น้ำหนักบรรทุกใช้งานต่อต้น P (kN)", min_value=50.0, max_value=10000.0, value=950.0, step=50.0)
    FS_req = st.number_input("Factor of Safety ที่ต้องการ (FS)", min_value=1.5, max_value=4.0, value=2.5, step=0.1)
    cov_soil = st.slider("ความแปรปรวนของชั้นดิน (CoV Soil)", 0.05, 0.40, 0.20, 0.05)

# ==========================================
# 7. MAIN AREA: BORING LOG UPLOAD & EDITOR
# ==========================================
st.subheader("📋 1. ข้อมูลชั้นดินจากหลุมเจาะ (Boring Log Input & File Upload)")

col_up1, col_up2, col_up3 = st.columns([2, 2, 1.5])
with col_up1:
    preset = st.selectbox("เลือกโปรไฟล์ชั้นดินตัวอย่าง:", [
        "โปรไฟล์ A: ดินเหนียวปนทรายโคราช (Nakhon Ratchasima)",
        "โปรไฟล์ B: ดินเหนียวอ่อนกรุงเทพฯ (Bangkok Soft Clay)"
    ])
with col_up2:
    uploaded_file = st.file_uploader("หรืออัปโหลดไฟล์ Boring Log (.csv / .xlsx)", type=["csv", "xlsx"])
with col_up3:
    sample_csv = pd.DataFrame({
        'Top (m)': [0.0, 3.0, 7.0, 12.0, 18.0],
        'Bottom (m)': [3.0, 7.0, 12.0, 18.0, 25.0],
        'Soil Type': ['Clay', 'Sand', 'Clay', 'Sand', 'Sand'],
        'SPT_N': [6, 14, 18, 28, 42],
        'Unit Weight (kN/m3)': [17.5, 18.5, 19.0, 19.8, 20.5]
    }).to_csv(index=False).encode('utf-8')
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button("📥 โหลดไฟล์ต้นแบบ CSV", data=sample_csv, file_name="boring_log_template.csv", mime="text/csv")

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            default_data = pd.read_csv(uploaded_file)
        else:
            default_data = pd.read_excel(uploaded_file)
        st.success(f"อัปโหลดไฟล์ `{uploaded_file.name}` สำเร็จ!")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
        default_data = pd.DataFrame()
else:
    if "โปรไฟล์ A" in preset:
        default_data = pd.DataFrame({
            'Top (m)': [0.0, 3.0, 7.0, 12.0, 18.0],
            'Bottom (m)': [3.0, 7.0, 12.0, 18.0, 25.0],
            'Soil Type': ['Clay', 'Sand', 'Clay', 'Sand', 'Sand'],
            'SPT_N': [6, 14, 18, 28, 42],
            'Unit Weight (kN/m3)': [17.5, 18.5, 19.0, 19.8, 20.5]
        })
    else:
        default_data = pd.DataFrame({
            'Top (m)': [0.0, 2.0, 11.0, 16.0, 22.0],
            'Bottom (m)': [2.0, 11.0, 16.0, 22.0, 30.0],
            'Soil Type': ['Clay', 'Clay', 'Clay', 'Sand', 'Sand'],
            'SPT_N': [4, 2, 12, 25, 38],
            'Unit Weight (kN/m3)': [16.5, 15.8, 18.2, 19.5, 20.2]
        })

df_soil = st.data_editor(
    default_data,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Soil Type": st.column_config.SelectboxColumn("Soil Type", options=["Clay", "Sand"], required=True),
        "SPT_N": st.column_config.NumberColumn("SPT-N (blows/ft)", min_value=0, max_value=100, required=True)
    }
)

# ==========================================
# 8. RUN SINGLE PILE SIMULATION
# ==========================================
sim_results = run_load_transfer_simulation(df_soil, D, L, Ep_GPa, gwt_depth)
Q_ult = sim_results['Q_ult']
Q_allow = Q_ult / FS_req
FS_actual = Q_ult / P_design
settlement_at_P = np.interp(P_design, sim_results['top_loads'], sim_results['top_settlements'])

R_samples, Q_samples, Pf, beta = run_monte_carlo_reliability(
    sim_results['Qs_max'], sim_results['Qb_max'], P_design, cov_soil=cov_soil
)

st.divider()
st.subheader("📊 2. สรุปผลการวิเคราะห์เสาเข็มเดี่ยว (Single Pile Performance)")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("กำลังประลัยรวม (Q_ult)", f"{Q_ult:,.1f} kN", f"Skin: {sim_results['Qs_max']:,.0f} | Tip: {sim_results['Qb_max']:,.0f}")
m2.metric("กำลังรับน้ำหนักปลอดภัย (Q_allow)", f"{Q_allow:,.1f} kN", f"FS = {FS_req:.1f}")
m3.metric("FS จริงของระบบ", f"{FS_actual:.2f}", "ผ่านเกณฑ์" if FS_actual >= FS_req else "ไม่ผ่านเกณฑ์!", delta_color="normal" if FS_actual >= FS_req else "inverse")
m4.metric("การทรุดตัวที่น้ำหนักใช้งาน", f"{settlement_at_P:.2f} mm", "ปกติ (< 25 mm)" if settlement_at_P < 25 else "ทรุดตัวสูง!", delta_color="normal" if settlement_at_P < 25 else "inverse")
m5.metric("โอกาสวิบัติ (Pf / Reliability β)", f"{Pf:.2f}%", f"β = {beta:.2f}", delta_color="normal" if Pf < 1.0 else "inverse")

# ==========================================
# 9. AI DIAGNOSTIC & RECOMMENDATION PANEL
# ==========================================
with st.expander("🤖 AI Geotechnical Diagnostic & Auto-Optimizer (คลิกดูคำแนะนำและขนาดที่เหมาะสมที่สุด)", expanded=True):
    insights, warnings, best_design = ai_soil_diagnostic(df_soil, D, L, P_design, FS_req, sim_results, settlement_at_P, Pf)
    
    col_ai1, col_ai2 = st.columns([2, 1])
    with col_ai1:
        for w in warnings:
            st.warning(w)
        for ins in insights:
            st.info(ins)
    with col_ai2:
        st.markdown("#### 🎯 AI Smart Sizing Recommendation")
        if best_design:
            st.success(f"""
            **ขนาดที่เหมาะสมและประหยัดที่สุด:**
            * **เส้นผ่านศูนย์กลาง (D):** `{best_design['D']:.2f} m`
            * **ความยาวเสาเข็ม (L):** `{best_design['L']:.1f} m`
            * **กำลังประลัย (Q_ult):** `{best_design['Q_ult']:,.0f} kN`
            * **การทรุดตัวคาดการณ์:** `{best_design['settle']:.1f} mm`
            """)
        else:
            st.error("ไม่พบขนาดเสาเข็มเดี่ยวที่ผ่านเกณฑ์ในระยะความลึกข้อมูลนี้ แนะนำให้ใช้ระบบกลุ่มเสาเข็ม (Pile Group) ด้านล่าง หรือเพิ่มความลึกหลุมเจาะ")

# ==========================================
# 10. TABS FOR SIMULATOR & PILE CAP DESIGN
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "📈 กราฟจำลองการถ่ายแรง (t-z / q-z Simulator)",
    "🏗️ ออกแบบกลุ่มเสาเข็มและฐานราก (Pile Cap SDM)",
    "📄 ส่งออกรายงานรายการคำนวณ (Export Report)"
])

with tab1:
    load_step_idx = st.slider(
        "เลื่อนแถบเพื่อจำลองการเพิ่มน้ำหนักบรรทุกที่หัวเสาเข็ม (Step Loading Simulation):",
        min_value=1, max_value=len(sim_results['top_loads']) - 1, value=15
    )
    selected_load = sim_results['top_loads'][load_step_idx]
    selected_settle = sim_results['top_settlements'][load_step_idx]
    depths_prof, loads_prof = sim_results['profiles'][load_step_idx]

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=sim_results['top_loads'], y=sim_results['top_settlements'], mode='lines', name='Load-Settlement', line=dict(color='#1f77b4', width=3)))
        fig1.add_trace(go.Scatter(x=[selected_load], y=[selected_settle], mode='markers+text', name='Current Step', text=[f"  {selected_load:,.0f} kN ({selected_settle:.1f} mm)"], textposition="middle right", marker=dict(color='red', size=12)))
        fig1.add_vline(x=P_design, line_dash="dash", line_color="green", annotation_text=f"Design Load ({P_design:.0f} kN)")
        fig1.update_layout(title="1. กราฟน้ำหนักบรรทุก vs การทรุดตัว (Load-Settlement)", xaxis_title="Axial Load at Pile Head (kN)", yaxis_title="Settlement (mm)", yaxis=dict(autorange="reversed"), height=420)
        st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=loads_prof, y=depths_prof, mode='lines+markers', name='Axial Force', fill='tozerox', line=dict(color='#ff7f0e', width=3)))
        fig2.update_layout(title=f"2. กราฟการถ่ายแรงตามความลึก (ที่แรง {selected_load:,.0f} kN)", xaxis_title="Axial Force in Pile (kN)", yaxis_title="Depth (m)", yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, max(sim_results['top_loads']) * 1.1]), height=420)
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=R_samples, name='Pile Capacity (R)', opacity=0.65, marker_color='#2ca02c', nbinsx=40))
    fig3.add_trace(go.Histogram(x=Q_samples, name='Applied Load (Q)', opacity=0.65, marker_color='#d62728', nbinsx=40))
    fig3.update_layout(barmode='overlay', title=f"3. การวิเคราะห์ความเสี่ยง Monte Carlo 2,000 รอบ (Pf = {Pf:.2f}%, β = {beta:.2f})", xaxis_title="Force (kN)", yaxis_title="Frequency", height=320)
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.markdown("### 🏗️ ออกแบบกลุ่มเสาเข็มและฐานรากหัวเสาเข็ม (วิธีหน่วยแรงประลัย SDM - ว.ส.ท.)")
    c1, c2, c3, c4 = st.columns(4)
    P_col_service = c1.number_input("แรงตามแนวแกนลงตอม่อ P_col (kN)", min_value=100.0, max_value=50000.0, value=2800.0, step=100.0)
    Mx_service = c2.number_input("โมเมนต์รอบแกน X, Mx (kN-m)", min_value=0.0, max_value=5000.0, value=120.0, step=10.0)
    My_service = c3.number_input("โมเมนต์รอบแกน Y, My (kN-m)", min_value=0.0, max_value=5000.0, value=80.0, step=10.0)
    fc_ksc = c4.selectbox("กำลังอัดคอนกรีต f'c (ksc)", [210, 240, 280, 320], index=1)

    cap_res = design_pile_cap_sdm(P_col_service, Mx_service, My_service, Q_allow, D, fc_ksc=fc_ksc)

    col_cap1, col_cap2 = st.columns([1.2, 1])
    with col_cap1:
        st.success(f"**ผลการออกแบบฐานราก:** ใช้เสาเข็มจำนวน **{cap_res['n_piles']} ต้น** (ระยะห่าง S = {cap_res['spacing']:.2f} m)")
        st.table(pd.DataFrame({
            "รายการออกแบบ (Design Item)": [
                "ขนาดฐานราก (กว้าง B × ยาว L × หนา H)",
                "ประสิทธิภาพกลุ่มเสาเข็ม (Group Efficiency ηg)",
                "กำลังรับน้ำหนักปลอดภัยต่อต้นในกลุ่ม",
                "แรงสูงสุดที่กระทำต่อเสาเข็ม 1 ต้น (Max Pile Load)",
                "พื้นที่เหล็กเสริมที่ต้องการ (As_req)",
                "การจัดเหล็กเสริมหลัก (แต่ละทิศทาง)"
            ],
            "ผลลัพธ์การคำนวณ": [
                f"{cap_res['B_cap']:.2f} m × {cap_res['L_cap']:.2f} m × {cap_res['H_cap']:.2f} m",
                f"{cap_res['eta_g']:.3f}",
                f"{cap_res['Q_allow_pile_eff']:,.1f} kN",
                f"{cap_res['max_pile_load']:,.1f} kN ({'ผ่าน' if cap_res['max_pile_load'] <= cap_res['Q_allow_pile_eff'] else 'เกินกำลัง!'})",
                f"{cap_res['As_req_cm2']:.2f} cm²",
                f"{cap_res['n_bars']}-DB20 @ {cap_res['bar_spacing_cm']:.1f} cm"
            ]
        }))

    with col_cap2:
        # วาดแปลน 2D Pile Cap Layout
        fig_cap = go.Figure()
        B_half = cap_res['B_cap'] / 2.0
        L_half = cap_res['L_cap'] / 2.0
        # ขอบฐานราก
        fig_cap.add_shape(type="rect", x0=-L_half, y0=-B_half, x1=L_half, y1=B_half, line=dict(color="black", width=3), fillcolor="rgba(200, 200, 200, 0.2)")
        # เสาตอม่อตรงกลาง
        fig_cap.add_shape(type="rect", x0=-0.25, y0=-0.25, x1=0.25, y1=0.25, line=dict(color="blue", width=2), fillcolor="rgba(0, 100, 255, 0.3)")
        # วงกลมเสาเข็มแต่ละต้น
        for idx, ((px, py), pload) in enumerate(zip(cap_res['coords'], cap_res['pile_loads'])):
            fig_cap.add_shape(type="circle", x0=px-D/2, y0=py-D/2, x1=px+D/2, y1=py+D/2, line=dict(color="darkred", width=2), fillcolor="rgba(255, 100, 100, 0.4)")
            fig_cap.add_annotation(x=px, y=py, text=f"P{idx+1}<br>{pload:.0f}kN", showarrow=False, font=dict(size=10, color="black"))
            
        max_dim = max(cap_res['B_cap'], cap_res['L_cap']) * 0.65
        fig_cap.update_layout(
            title=f"แปลนจัดวางเสาเข็ม ({cap_res['n_piles']} ต้น) และขนาดฐานราก",
            xaxis=dict(range=[-max_dim, max_dim], title="X (m)", scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-max_dim, max_dim], title="Y (m)"),
            height=400, margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_cap, use_container_width=True)

with tab3:
    st.markdown("### 📄 ส่งออกรายการคำนวณวิศวกรรม (Engineering Calculation Report)")
    st.info("กดปุ่มด้านล่างเพื่อดาวน์โหลดไฟล์รายการคำนวณฉบับสมบูรณ์ สามารถเปิดในเบราว์เซอร์และกด **Ctrl+P -> Save as PDF** เพื่อแนบส่งเป็นภาคผนวกในเล่มโปรเจกต์ได้ทันที")
    
    report_html = generate_html_report(project_name, D, L, Ep_GPa, gwt_depth, P_design, FS_req, df_soil, sim_results, settlement_at_P, Pf, beta, cap_res)
    st.download_button(
        label="📥 ดาวน์โหลดรายการคำนวณ (Calculation Report .html / Print-to-PDF)",
        data=report_html.encode('utf-8'),
        file_name=f"Pile_Design_Report_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html"
    )
