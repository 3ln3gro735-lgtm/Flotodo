# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import os
import time 
from collections import defaultdict, Counter
import unicodedata

# --- CONFIGURACION DE LA RUTA ---
RUTA_CSV = 'Flotodo.csv'
RUTA_CACHE = 'cache_perfiles_florida.csv'

# --- CONFIGURACION DE LA PAGINA ---
st.set_page_config(
    page_title="Florida - Análisis de Sorteos",
    page_icon="🌴",
    layout="wide"
)

st.title("🌴 Florida - Análisis de Sorteos")
st.markdown("Visualización de Estado Actual (Solo Tarde y Noche).")

# --- FUNCIONES AUXILIARES Y DE CARGA ---

def remove_accents(input_str):
    if not isinstance(input_str, str): return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def inicializar_archivo(ruta, columnas):
    if not os.path.exists(ruta):
        try:
            with open(ruta, 'w', encoding='latin-1') as f:
                f.write(";".join(columnas) + "\n")
        except Exception as e:
            st.error(f"Error inicializando {ruta}: {e}")

@st.cache_data(ttl="10m") 
def cargar_datos_flotodo(_ruta_csv):
    try:
        inicializar_archivo(_ruta_csv, ["Fecha","Tipo_Sorteo","Centena","Fijo","Primer_Corrido","Segundo_Corrido"])
        
        try:
            df = pd.read_csv(_ruta_csv, sep=';', encoding='latin-1', header=0, on_bad_lines='skip', dtype=str)
        except Exception as e:
            st.error(f"Error leyendo CSV: {e}")
            return pd.DataFrame()

        if df.empty: return pd.DataFrame()
        
        df.columns = [str(c).strip() for c in df.columns]
        
        rename_map = {}
        for col in df.columns:
            c = str(col).strip()
            if 'Fecha' in c: rename_map[col] = 'Fecha'
            elif 'Noche' in c or 'Tarde' in c: rename_map[col] = 'Tipo_Sorteo'
            elif 'Centena' in c: rename_map[col] = 'Centena'
            elif 'Fijo' in c and 'Corrido' not in c: rename_map[col] = 'Fijo'
            elif '1er' in c or 'Primer' in c: rename_map[col] = 'Primer_Corrido'
            elif '2do' in c or 'Segundo' in c: rename_map[col] = 'Segundo_Corrido'
        
        df.rename(columns=rename_map, inplace=True)
        
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Fecha'], inplace=True)
        
        # Mapeo solo para Tarde y Noche
        df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper().map({
            'TARDE': 'T', 'T': 'T', 
            'NOCHE': 'N', 'N': 'N'
        }).fillna('OTRO')
        df = df[df['Tipo_Sorteo'].isin(['T', 'N'])].copy()
        
        for col in ['Centena', 'Fijo', 'Primer_Corrido', 'Segundo_Corrido']:
            if col not in df.columns: df[col] = '0'
            df[col] = df[col].replace('', '0').fillna('0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        df_long = df.melt(id_vars=['Fecha', 'Tipo_Sorteo'], 
                          value_vars=['Centena', 'Fijo', 'Primer_Corrido', 'Segundo_Corrido'],
                          var_name='Posicion', value_name='Numero')
        
        pos_map = {'Centena': 'Centena', 'Fijo': 'Fijo', 
                   'Primer_Corrido': '1er Corrido', 'Segundo_Corrido': '2do Corrido'}
        df_long['Posicion'] = df_long['Posicion'].map(pos_map)
        
        df_historial = df_long.dropna(subset=['Numero']).copy()
        df_historial['Numero'] = df_historial['Numero'].astype(int)
        
        # Ordenamiento Cronológico: Tarde(0), Noche(1)
        draw_order_map = {'T': 0, 'N': 1}
        df_historial['draw_order'] = df_historial['Tipo_Sorteo'].map(draw_order_map)
        df_historial['sort_key'] = df_historial['Fecha'] + pd.to_timedelta(df_historial['draw_order'], unit='h')
        df_historial = df_historial.sort_values(by='sort_key').reset_index(drop=True)
        df_historial.drop(columns=['draw_order', 'sort_key'], inplace=True)
        
        return df_historial
    except Exception as e:
        st.error(f"Error crítico procesando datos: {str(e)}")
        return pd.DataFrame()

def calcular_estado_actual(gap, limite_dinamico):
    if pd.isna(limite_dinamico) or limite_dinamico == 0: return "Normal"
    if gap > limite_dinamico: return "Muy Vencido"
    elif gap > (limite_dinamico * 0.66): return "Vencido"
    else: return "Normal"

def obtener_df_temperatura(contador):
    df = pd.DataFrame.from_dict(contador, orient='index', columns=['Frecuencia'])
    df = df.reset_index().rename(columns={'index': 'Dígito'})
    df = df.sort_values('Frecuencia', ascending=False).reset_index(drop=True)
    df['Temperatura'] = '🧊 Frío'
    if len(df) >= 3: df.loc[0:2, 'Temperatura'] = '🔥 Caliente'
    if len(df) >= 7: df.loc[6:9, 'Temperatura'] = '🧊 Frío'
    if len(df) >= 3: df.loc[3:5, 'Temperatura'] = '🟡 Tibio'
    return df

# --- FUNCIONES DE ANALISIS ---

def analizar_oportunidad_por_digito(df_historial, fecha_referencia):
    if df_historial.empty: return pd.DataFrame(), pd.DataFrame()
    df_base_fijos = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    
    contador_decenas = Counter()
    contador_unidades = Counter()
    for num in df_base_fijos['Numero']:
        contador_decenas[num // 10] += 1
        contador_unidades[num % 10] += 1

    df_temp_dec = obtener_df_temperatura(contador_decenas)
    df_temp_uni = obtener_df_temperatura(contador_unidades)
    
    mapa_temp_dec = pd.Series(df_temp_dec.Temperatura.values, index=df_temp_dec.Dígito).to_dict()
    mapa_temp_uni = pd.Series(df_temp_uni.Temperatura.values, index=df_temp_uni.Dígito).to_dict()
    
    df_hist_estado = df_base_fijos[df_base_fijos['Fecha'] < fecha_referencia].copy()
    
    res_dec, res_uni = [], []
    for i in range(10):
        fechas_d = df_hist_estado[df_hist_estado['Numero'] // 10 == i]['Fecha'].sort_values()
        gap_d, prom_d = 0, 0
        if not fechas_d.empty:
            gaps = fechas_d.diff().dt.days.dropna()
            prom_d = gaps.median() if len(gaps) > 0 else 0
            gap_d = (fecha_referencia - fechas_d.max()).days
        ed = calcular_estado_actual(gap_d, prom_d)
        
        fechas_u = df_hist_estado[df_hist_estado['Numero'] % 10 == i]['Fecha'].sort_values()
        gap_u, prom_u = 0, 0
        if not fechas_u.empty:
            gaps = fechas_u.diff().dt.days.dropna()
            prom_u = gaps.median() if len(gaps) > 0 else 0
            gap_u = (fecha_referencia - fechas_u.max()).days
        eu = calcular_estado_actual(gap_u, prom_u)
        
        p_base_d = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}[ed]
        p_base_u = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}[eu]
        
        res_dec.append({'Dígito': i, 'Temperatura': mapa_temp_dec.get(i, '🟡 Tibio'), 'Estado': ed, 'Punt. Base': p_base_d})
        res_uni.append({'Dígito': i, 'Temperatura': mapa_temp_uni.get(i, '🟡 Tibio'), 'Estado': eu, 'Punt. Base': p_base_u})

    return pd.DataFrame(res_dec), pd.DataFrame(res_uni)

# --- GESTOR DE CACHE ---
def obtener_historial_perfiles_cacheado(df_full, ruta_cache=None):
    if df_full.empty: return pd.DataFrame()
    
    df_fijos = df_full[df_full['Posicion'] == 'Fijo'].copy()
    df_cache = pd.DataFrame()
    
    use_file = ruta_cache and os.path.exists(ruta_cache)
    
    if use_file:
        try:
            df_cache = pd.read_csv(ruta_cache, parse_dates=['Fecha'])
        except:
            df_cache = pd.DataFrame() 

    df_fijos['ID_Sorteo'] = df_fijos['Fecha'].astype(str) + "_" + df_fijos['Tipo_Sorteo']
    
    ids_en_cache = set()
    if not df_cache.empty:
        sorteo_map_inv = {'Noche': 'N', 'Tarde': 'T'}
        df_cache['ID_Sorteo'] = df_cache['Fecha'].astype(str) + "_" + df_cache['Sorteo'].map(sorteo_map_inv)
        ids_en_cache = set(df_cache['ID_Sorteo'])
    
    df_nuevos = df_fijos[~df_fijos['ID_Sorteo'].isin(ids_en_cache)].copy()
    
    if df_nuevos.empty:
        if 'ID_Sorteo' in df_cache.columns: df_cache.drop(columns=['ID_Sorteo'], inplace=True)
        return df_cache
    
    df_nuevos = df_nuevos.sort_values(by=['Fecha', 'Tipo_Sorteo'])

    hist_decenas = defaultdict(list)
    hist_unidades = defaultdict(list)
    
    if not df_cache.empty:
        # Orden T(0), N(1)
        sort_val_inv = {'Tarde': 0, 'Noche': 1}
        df_cache['sort_val'] = df_cache['Sorteo'].map(sort_val_inv)
        df_cache_sorted = df_cache.sort_values(by=['Fecha', 'sort_val'])
        for _, row in df_cache_sorted.iterrows():
            num = int(row['Numero'])
            fecha = row['Fecha']
            hist_decenas[num // 10].append(fecha)
            hist_unidades[num % 10].append(fecha)
            
    nuevos_registros = []
    
    for idx, row in df_nuevos.iterrows():
        fecha_actual = row['Fecha']
        num_actual = row['Numero']
        tipo_actual = row['Tipo_Sorteo']
        
        dec = num_actual // 10
        uni = num_actual % 10
        
        fechas_dec_ant = [f for f in hist_decenas[dec] if f < fecha_actual]
        if fechas_dec_ant:
            last_dec = max(fechas_dec_ant)
            gap_dec = (fecha_actual - last_dec).days
            sorted_fds = sorted(fechas_dec_ant)
            gaps_d = [(sorted_fds[i] - sorted_fds[i-1]).days for i in range(1, len(sorted_fds))]
            med_d = np.median(gaps_d) if gaps_d else 0
            estado_dec = calcular_estado_actual(gap_dec, med_d)
        else:
            estado_dec = "Normal"
            
        fechas_uni_ant = [f for f in hist_unidades[uni] if f < fecha_actual]
        if fechas_uni_ant:
            last_uni = max(fechas_uni_ant)
            gap_uni = (fecha_actual - last_uni).days
            sorted_fus = sorted(fechas_uni_ant)
            gaps_u = [(sorted_fus[i] - sorted_fus[i-1]).days for i in range(1, len(sorted_fus))]
            med_u = np.median(gaps_u) if gaps_u else 0
            estado_uni = calcular_estado_actual(gap_uni, med_u)
        else:
            estado_uni = "Normal"
            
        perfil = f"{estado_dec}-{estado_uni}"
        nombre_sorteo = {'T': 'Tarde', 'N': 'Noche'}.get(tipo_actual, 'Otro')
        
        nuevos_registros.append({
            'Fecha': fecha_actual,
            'Sorteo': nombre_sorteo,
            'Numero': num_actual,
            'Perfil': perfil
        })
        
        hist_decenas[dec].append(fecha_actual)
        hist_unidades[uni].append(fecha_actual)
    
    if nuevos_registros:
        df_nuevos_cache = pd.DataFrame(nuevos_registros)
        if not df_cache.empty:
            cols_to_drop = [c for c in ['ID_Sorteo', 'sort_val'] if c in df_cache.columns]
            df_final = pd.concat([df_cache.drop(columns=cols_to_drop, errors='ignore'), df_nuevos_cache], ignore_index=True)
        else:
            df_final = df_nuevos_cache
            
        if ruta_cache:
            df_final.to_csv(ruta_cache, index=False)
        return df_final
    else:
        if 'ID_Sorteo' in df_cache.columns: df_cache.drop(columns=['ID_Sorteo'], inplace=True)
        return df_cache

def calcular_estabilidad_historica_digitos(df_full):
    if df_full.empty: return pd.DataFrame()
    df_fijos = df_full[df_full['Posicion'] == 'Fijo'].copy()
    resultados = []
    
    for i in range(10):
        fechas_d = df_fijos[df_fijos['Numero'] // 10 == i]['Fecha'].sort_values()
        if len(fechas_d) > 1:
            gaps = fechas_d.diff().dt.days.dropna()
            med = gaps.median()
            excesos = sum(g > (med * 1.5) for g in gaps)
            estabilidad = 100 - (excesos / len(gaps) * 100)
        else:
            estabilidad = 50
        resultados.append({'Digito': i, 'Tipo': 'Decena', 'EstabilidadHist': estabilidad})
        
        fechas_u = df_fijos[df_fijos['Numero'] % 10 == i]['Fecha'].sort_values()
        if len(fechas_u) > 1:
            gaps = fechas_u.diff().dt.days.dropna()
            med = gaps.median()
            excesos = sum(g > (med * 1.5) for g in gaps)
            estabilidad = 100 - (excesos / len(gaps) * 100)
        else:
            estabilidad = 50
        resultados.append({'Digito': i, 'Tipo': 'Unidad', 'EstabilidadHist': estabilidad})
        
    return pd.DataFrame(resultados)

# --- ANALISIS ESTADISTICAS PERFILES (P75) ---
def analizar_estadisticas_perfiles(df_historial_perfiles, fecha_referencia):
    historial_fechas_perfiles = defaultdict(list)
    ultimo_suceso_perfil = {}
    transiciones = Counter()
    ultimo_perfil_global = None
    
    sort_val = {'Tarde': 0, 'Noche': 1}
    df_historial_perfiles = df_historial_perfiles.copy()
    df_historial_perfiles['sort_val'] = df_historial_perfiles['Sorteo'].map(sort_val)
    df_historial_perfiles = df_historial_perfiles.sort_values(by=['Fecha', 'sort_val'])

    for _, row in df_historial_perfiles.iterrows():
        perfil = row['Perfil']
        fecha = row['Fecha']
        numero = row['Numero']
        
        historial_fechas_perfiles[perfil].append(fecha)
        ultimo_suceso_perfil[perfil] = row
        
        if ultimo_perfil_global:
            transiciones[(ultimo_perfil_global, perfil)] += 1
        ultimo_perfil_global = perfil
    
    total_salidas_perfil = Counter()
    for (origen, destino), count in transiciones.items():
        total_salidas_perfil[origen] += count
        
    analisis_perfiles = []
    
    for perfil, fechas in historial_fechas_perfiles.items():
        fechas_ordenadas = sorted(fechas)
        ultima_fecha = fechas_ordenadas[-1]
        
        gaps = []
        for k in range(1, len(fechas_ordenadas)):
            gaps.append((fechas_ordenadas[k] - fechas_ordenadas[k-1]).days)
            
        mediana_gap_actual = np.median(gaps) if gaps else 0
        gap_actual = (fecha_referencia - ultima_fecha).days
        
        if len(gaps) >= 4:
            limite_dinamico = int(np.percentile(gaps, 75))
        elif len(gaps) > 0:
            limite_dinamico = int(mediana_gap_actual * 2)
        else:
            limite_dinamico = 0

        estado_actual = calcular_estado_actual(gap_actual, limite_dinamico)
        
        estados_historicos = [calcular_estado_actual(g, limite_dinamico) for g in gaps] if gaps else []
        total_hist = len(estados_historicos)
        count_normal = estados_historicos.count('Normal')
        count_vencido = estados_historicos.count('Vencido')
        count_muy_vencido = estados_historicos.count('Muy Vencido')
        
        muy_vencidos_count = count_muy_vencido
        estabilidad_actual = ((total_hist - muy_vencidos_count) / total_hist * 100) if total_hist > 0 else 0
        
        alerta_recuperacion = False
        if estabilidad_actual > 60 and estado_actual in ['Vencido', 'Muy Vencido']:
            alerta_recuperacion = True
        
        tiempo_limite = limite_dinamico
        
        repeticiones = transiciones.get((perfil, perfil), 0)
        total_salidas = total_salidas_perfil.get(perfil, 0)
        prob_repeticion = (repeticiones / total_salidas * 100) if total_salidas > 0 else 0
        
        semana_activa = "Sí" if estado_actual in ['Vencido', 'Muy Vencido'] else "No"
        
        last_row = ultimo_suceso_perfil[perfil]
        
        estado_ultima_salida = "Normal"
        estabilidad_ultima_salida = 0.0
        exceso_ultima_salida = 0
        
        if len(gaps) >= 1:
            gap_ultima_espera = gaps[-1]
            if len(gaps) > 1:
                gaps_prev = gaps[:-1]
                if len(gaps_prev) >= 4: lim_prev = int(np.percentile(gaps_prev, 75))
                else: lim_prev = int(np.median(gaps_prev) * 2)
                estado_ultima_salida = calcular_estado_actual(gap_ultima_espera, lim_prev)
                
                if estado_ultima_salida == "Muy Vencido":
                    exceso_ultima_salida = int(gap_ultima_espera - lim_prev)
                
                ests_prev = [calcular_estado_actual(g, lim_prev) for g in gaps_prev]
                mv_prev = ests_prev.count('Muy Vencido')
                estabilidad_ultima_salida = ((len(ests_prev) - mv_prev) / len(ests_prev) * 100)
            else:
                estado_ultima_salida = "Normal"
                estabilidad_ultima_salida = 100.0
        
        analisis_perfiles.append({
            'Perfil': perfil,
            'Frecuencia': total_hist + 1,
            'Última Fecha': ultima_fecha,
            'Gap Actual': gap_actual,
            'Mediana Gap': int(mediana_gap_actual),
            'Estado Actual': estado_actual,
            'Estabilidad': round(estabilidad_actual, 1),
            'Tiempo Limite': tiempo_limite,
            'Alerta': '⚠️ RECUPERAR' if alerta_recuperacion else '-',
            'Prob Repeticiones %': round(prob_repeticion, 1),
            'Semana Activa': semana_activa,
            'Último Numero': last_row['Numero'],
            'Último Sorteo': last_row['Sorteo'],
            'Veces Normal': count_normal,
            'Veces Vencido': count_vencido,
            'Veces Muy Vencido': count_muy_vencido,
            'Estado Ultima Salida': estado_ultima_salida,
            'Estabilidad Ultima Salida': round(estabilidad_ultima_salida, 1),
            'Exceso Ultima Salida': exceso_ultima_salida
        })
        
    df_stats = pd.DataFrame(analisis_perfiles)
    return df_stats, transiciones, ultimo_perfil_global

# --- MOTORES DE PREDICCION ---
def obtener_prediccion_numeros_lista(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos):
    scores = []
    
    map_est_dec = estabilidad_digitos[(estabilidad_digitos['Tipo']=='Decena')].set_index('Digito')['EstabilidadHist'].to_dict()
    map_est_uni = estabilidad_digitos[(estabilidad_digitos['Tipo']=='Unidad')].set_index('Digito')['EstabilidadHist'].to_dict()
    
    for _, row in df_stats.iterrows():
        p = row['Perfil']
        score = 0
        estado = row['Estado Actual']
        
        if row['Alerta'] == '⚠️ RECUPERAR': score += 150 
        else:
            if estado == 'Vencido': score += 70 
            elif estado == 'Normal': score += 50 
            elif estado == 'Muy Vencido': score += 30 
        
        score += row['Estabilidad'] * 0.5
        trans_count = transizioni.get((ultimo_perfil, p), 0)
        score += trans_count * 10 
        
        scores.append({'Perfil': p, 'Score': int(score), 'Estado': estado})
    
    df_scores = pd.DataFrame(scores).sort_values('Score', ascending=False)
    top_3 = df_scores.head(3)
    
    map_estado_dec = df_oport_dec.set_index('Dígito')['Estado'].to_dict()
    map_estado_uni = df_oport_uni.set_index('Dígito')['Estado'].to_dict()
    
    df_hist_nums = df_historial_perfiles.groupby('Numero')['Fecha'].max()
    candidatos_totales = []
    
    map_temp_dec = df_oport_dec.set_index('Dígito')['Temperatura'].to_dict()
    map_temp_uni = df_oport_uni.set_index('Dígito')['Temperatura'].to_dict()
    temp_val = {'🔥 Caliente': 3, '🟡 Tibio': 2, '🧊 Frío': 1}
    
    for _, row in top_3.iterrows():
        perfil = row['Perfil']
        partes = perfil.split('-')
        ed_req, eu_req = partes[0], partes[1]
        
        decenas_estado = [d for d in range(10) if map_estado_dec.get(d) == ed_req]
        unidades_estado = [u for u in range(10) if map_estado_uni.get(u) == eu_req]
        
        for d in decenas_estado:
            for u in unidades_estado:
                num = int(f"{d}{u}")
                last_seen = df_hist_nums.get(num, pd.Timestamp('2000-01-01'))
                gap_n = (fecha_ref - last_seen).days if isinstance(last_seen, pd.Timestamp) else 999
                
                temp_d = temp_val.get(map_temp_dec.get(d, '🟡 Tibio'), 2)
                temp_u = temp_val.get(map_temp_uni.get(u, '🟡 Tibio'), 2)
                temp_score = temp_d + temp_u
                
                est_d = map_est_dec.get(d, 50)
                est_u = map_est_uni.get(u, 50)
                bonus_est = (est_d + est_u) / 20 
                
                candidatos_totales.append({
                    'Numero': num, 'Perfil': perfil, 'Score': row['Score'], 
                    'Gap_Num': gap_n, 'Temp_Score': temp_score + bonus_est
                })
                
    df_cands = pd.DataFrame(candidatos_totales)
    if df_cands.empty: return []
    df_cands = df_cands.sort_values(['Score', 'Temp_Score', 'Gap_Num', 'Numero'], ascending=[False, False, False, True]).drop_duplicates(subset=['Numero'])
    return df_cands.head(30)['Numero'].tolist()

def generar_sugerencia_fusionada(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos):
    st.subheader("🤖 Sugerencia Inteligente Fusionada")
    
    with st.expander("📖 ¿Cómo funciona la lógica de 'Muy Vencido'?"):
        st.markdown("""
        **Definición de Estados (P75):**
        - **Normal:** El dígito ha salido hace poco.
        - **Vencido:** El dígito ha superado su promedio.
        - **Muy Vencido:** El dígito ha superado el **Percentil 75 (P75)**.
        """)

    st.markdown("### 🚨 Detalle de Alertas Activas")
    
    map_estado_dec = df_oport_dec.set_index('Dígito')['Estado'].to_dict()
    map_estado_uni = df_oport_uni.set_index('Dígito')['Estado'].to_dict()
    
    alertas_activas = df_stats[df_stats['Alerta'] == '⚠️ RECUPERAR'].copy()
    
    if not alertas_activas.empty:
        for _, row_alert in alertas_activas.iterrows():
            perfil_name = row_alert['Perfil']
            partes = perfil_name.split('-')
            ed_req, eu_req = partes[0], partes[1]
            
            decenas_cumplen = [d for d in range(10) if map_estado_dec.get(d) == ed_req]
            unidades_cumplen = [u for u in range(10) if map_estado_uni.get(u) == eu_req]
            
            nums_alerta = []
            for d in decenas_cumplen:
                for u in unidades_cumplen:
                    val = int(f"{d}{u}")
                    nums_alerta.append(f"{val:02d}")
            
            gap = row_alert['Gap Actual']
            med = row_alert['Mediana Gap']
            estado = row_alert['Estado Actual']
            tiempo_limite = row_alert['Tiempo Limite']
            
            ult_fecha = row_alert['Última Fecha']
            ult_estado = row_alert['Estado Ultima Salida']
            ult_estabilidad = row_alert['Estabilidad Ultima Salida']
            ult_exceso = row_alert['Exceso Ultima Salida']
            
            time_str = ""
            if estado == "Normal":
                falta = int(med - gap)
                time_str = f"🟢 Faltan {falta} días"
            elif estado == "Vencido":
                falta_mv = int(tiempo_limite - gap)
                exceso = int(gap - med)
                time_str = f"🟠 Exceso {exceso} días"
            elif estado == "Muy Vencido":
                exceso = int(gap - tiempo_limite)
                time_str = f"🔴 +{exceso} días exceso"
            
            st.markdown(f"**Perfil Alertado: `{perfil_name}`**")
            st.markdown(f"📍 **Estado Actual:** `{estado}` | ⏳ {time_str}")
            st.markdown(f"📊 **Datos:** Gap: **{gap} días** | Límite P75: **{tiempo_limite} días**")
            
            st.markdown(f"Decenas: `{decenas_cumplen}` | Unidades: `{unidades_cumplen}`")
            
            if nums_alerta:
                st.success(f"🔢 Números que componen esta alerta ({len(nums_alerta)}):")
                chunks = [nums_alerta[x:x+10] for x in range(0, len(nums_alerta), 10)]
                for chunk in chunks:
                    st.write(f"`{' '.join(chunk)}`")
            
            st.info(f"🔙 **Historia de la última vez que salió este perfil:**")
            st.markdown(f"- Fecha: **{ult_fecha.strftime('%d/%m/%Y')}**")
            st.markdown(f"- Estado en ese momento: **{ult_estado}**")
            st.markdown(f"- Estabilidad previa: **{ult_estabilidad}%**")
            if ult_estado == "Muy Vencido":
                st.markdown(f"- Días en exceso: **{ult_exceso} días**")
            
            st.markdown("---")
    else:
        st.info("No hay alertas de recuperación activas.")

    st.markdown("### 🎲 Top 30 Números Sugeridos")
    lista_nums = obtener_prediccion_numeros_lista(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos)
    
    scores_display = []
    for _, row in df_stats.iterrows():
        p = row['Perfil']
        score = 0
        estado = row['Estado Actual']
        if row['Alerta'] == '⚠️ RECUPERAR': score += 150
        else:
            if estado == 'Vencido': score += 70 
            elif estado == 'Normal': score += 50 
            elif estado == 'Muy Vencido': score += 30
        score += row['Estabilidad'] * 0.5
        trans_count = transizioni.get((ultimo_perfil, p), 0)
        score += trans_count * 10 
        scores_display.append({'Perfil': p, 'Score': int(score), 'Estado': estado})
    
    df_scores = pd.DataFrame(scores_display).sort_values('Score', ascending=False)
    st.dataframe(df_scores.head(3), hide_index=True)
    
    if not lista_nums:
        st.warning("No se generaron candidatos.")
        return
    
    def get_state_color_hex(state):
        if state == 'Normal': return '#00FF00'
        elif state == 'Vencido': return '#FFFF00'
        elif state == 'Muy Vencido': return '#FF0000'
        return '#FFFFFF'

    def shorten_state(text):
        if text == "Muy Vencido": return "M. Vencido"
        return text

    cols = st.columns(6)
    for idx, num in enumerate(lista_nums):
        d_int = int(num // 10)
        u_int = int(num % 10)
        ed = map_estado_dec.get(d_int, "?")
        eu = map_estado_uni.get(u_int, "?")
        
        color_dec = get_state_color_hex(ed)
        color_uni = get_state_color_hex(eu)
        
        ed_display = shorten_state(ed)
        eu_display = shorten_state(eu)
        
        cols[idx % 6].markdown(f"""
        <div style="background-color:#000000; padding:10px; border-radius:8px; text-align:center; border: 1px solid #333;">
            <h2 style="margin:0; color:#00FF00; font-weight:bold;">{num:02d}</h2>
            <hr style="margin: 4px 0; border-top: 1px solid #444;">
            <div style="font-size: 0.85em; font-weight:bold; color:{color_dec};">{ed_display}</div>
            <div style="font-size: 0.85em; font-weight:bold; color:{color_uni};">{eu_display}</div>
        </div>
        """, unsafe_allow_html=True)

# --- BACKTEST ---
def ejecutar_backtest(df_full, dias_atras):
    hoy = datetime.now().date()
    resultados = []
    aciertos = 0
    total_sorteos = 0
    
    df_cache_full = obtener_historial_perfiles_cacheado(df_full)
    estabilidad_digitos = calcular_estabilidad_historica_digitos(df_full)
    
    progress_bar = st.progress(0)
    
    for i in range(dias_atras):
        current_date = hoy - timedelta(days=dias_atras - i - 1)
        fecha_ref = pd.to_datetime(current_date)
        
        for sorteo_tipo in ['T', 'N']:
            target_sesion_bt = {'T': 'Tarde', 'N': 'Noche'}[sorteo_tipo]
            target_val = {'T': 0, 'N': 1}[sorteo_tipo]
            
            mask_real = (df_full['Fecha'] == fecha_ref) & (df_full['Tipo_Sorteo'] == sorteo_tipo) & (df_full['Posicion'] == 'Fijo')
            if mask_real.sum() == 0: continue
            
            resultado_real = df_full[mask_real]['Numero'].iloc[0]
            total_sorteos += 1
            
            # Logica temporal para backtest
            if target_val == 0: # Target Tarde
                df_backtest = df_full[df_full['Fecha'] < fecha_ref].copy()
                df_historial_perfiles = df_cache_full[df_cache_full['Fecha'] < fecha_ref].copy()
            else: # Target Noche
                df_backtest = df_full[(df_full['Fecha'] < fecha_ref) | ((df_full['Fecha'] == fecha_ref) & (df_full['Tipo_Sorteo'] == 'T'))].copy()
                df_historial_perfiles = df_cache_full[(df_cache_full['Fecha'] < fecha_ref) | ((df_cache_full['Fecha'] == fecha_ref) & (df_cache_full['Sorteo'] == 'Tarde'))].copy()
            
            if df_backtest.empty or df_historial_perfiles.empty: continue
            
            df_oport_dec, df_oport_uni = analizar_oportunidad_por_digito(df_backtest, fecha_ref)
            df_stats, transizioni, ultimo_perfil = analizar_estadisticas_perfiles(df_historial_perfiles, fecha_ref)
            prediccion = obtener_prediccion_numeros_lista(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos)
            
            if resultado_real in prediccion: aciertos += 1
            
            resultados.append({
                'Fecha': current_date, 'Sorteo': target_sesion_bt,
                'Real': resultado_real, 'Resultado': '✅' if resultado_real in prediccion else '❌'
            })
        
        progress_bar.progress((i + 1) / dias_atras)
        
    progress_bar.empty()
    return pd.DataFrame(resultados), aciertos, total_sorteos

# --- MAIN ---
def main():
    st.sidebar.header("⚙️ Opciones")
    
    df_full = cargar_datos_flotodo(RUTA_CSV)
    
    # --- FORMULARIO AGREGAR SORTEO ---
    with st.sidebar.expander("📝 Agregar Sorteo", True):
        f_nueva = st.date_input("Fecha", datetime.now().date())
        ses = st.radio("Sesión", ["Tarde", "Noche"], horizontal=True)
        
        cent = st.number_input("Centena", 0, 999, 0, key="inp_cent")
        fij = st.number_input("Fijo", 0, 99, 0, key="inp_fijo")
        c1 = st.number_input("1er Corrido", 0, 99, 0, key="inp_c1")
        c2 = st.number_input("2do Corrido", 0, 99, 0, key="inp_c2")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("💾 Guardar Sorteo"):
                s_code_map = {"Tarde": "T", "Noche": "N"}
                s_code = s_code_map[ses]
                line = f"{f_nueva.strftime('%d/%m/%Y')};{s_code};{int(cent)};{int(fij)};{int(c1)};{int(c2)}\n"
                try:
                    with open(RUTA_CSV, 'a', encoding='latin-1') as file: file.write(line)
                    st.success("¡Guardado con éxito!")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                except Exception as err: st.error(f"Error: {err}")

        with col_btn2:
            if st.button("⏪ Deshacer Último"):
                try:
                    if os.path.exists(RUTA_CSV):
                        with open(RUTA_CSV, 'r', encoding='latin-1') as f: lines = f.readlines()
                        if len(lines) > 1:
                            lines.pop()
                            with open(RUTA_CSV, 'w', encoding='latin-1') as f: f.writelines(lines)
                            st.warning("Último sorteo eliminado.")
                            st.cache_data.clear()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("El archivo está vacío.")
                except Exception as err: st.error(f"Error: {err}")

    # --- LOGICA DE ANALISIS Y SIDEBAR (TARDE/NOCHE) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Últimos Resultados")
    
    fecha_ref_default = pd.Timestamp.now(tz=None).normalize()
    target_sesion_default = "Tarde"
    info_ultimo_sorteo = None
    
    if not df_full.empty:
        # Orden T(0), N(1)
        sort_order_map = {'T': 0, 'N': 1}
        df_sort = df_full[df_full['Posicion'] == 'Fijo'].copy()
        df_sort['order_val'] = df_sort['Tipo_Sorteo'].map(sort_order_map)
        df_fijos_sorted = df_sort.sort_values(by=['Fecha', 'order_val'], ascending=[True, True])
        ultimo_registro = df_fijos_sorted.iloc[-1]
        
        u_fecha = ultimo_registro['Fecha'].date()
        u_tipo = ultimo_registro['Tipo_Sorteo']
        
        ultimos = {}
        for tipo, label in [('T', 'Tarde'), ('N', 'Noche')]:
            df_tipo = df_full[df_full['Tipo_Sorteo'] == tipo]
            if not df_tipo.empty:
                last_row = df_tipo[df_tipo['Fecha'] == df_tipo['Fecha'].max()].iloc[0]
                ultimos[tipo] = last_row
        
        iconos = {'T': '☀️', 'N': '🌙'}
        
        for tipo in ['T', 'N']:
            if tipo in ultimos:
                row = ultimos[tipo]
                f_str = row['Fecha'].strftime('%d/%m/%Y')
                st.sidebar.markdown(f"**{iconos[tipo]} {row['Tipo_Sorteo']} ({f_str})**")
                
                mask = (df_full['Fecha'] == row['Fecha']) & (df_full['Tipo_Sorteo'] == tipo)
                try:
                    m_c = mask & (df_full['Posicion'] == 'Centena')
                    m_f = mask & (df_full['Posicion'] == 'Fijo')
                    m_1 = mask & (df_full['Posicion'] == '1er Corrido')
                    m_2 = mask & (df_full['Posicion'] == '2do Corrido')
                    
                    val_c = df_full.loc[m_c, 'Numero'].iloc[0] if m_c.any() else 0
                    val_f = df_full.loc[m_f, 'Numero'].iloc[0] if m_f.any() else 0
                    val_1 = df_full.loc[m_1, 'Numero'].iloc[0] if m_1.any() else 0
                    val_2 = df_full.loc[m_2, 'Numero'].iloc[0] if m_2.any() else 0
                    
                    num_completo = int(f"{val_c}{val_f:02d}")
                    
                    st.sidebar.markdown(f"Num: **{num_completo}**")
                    st.sidebar.markdown(f"C1: `{val_1}` | C2: `{val_2}`")
                except:
                    st.sidebar.markdown("Error datos")
                
                st.sidebar.markdown("---")
        
        fecha_ref_default = ultimo_registro['Fecha']
        target_sesion_default = {'T': 'Tarde', 'N': 'Noche'}[u_tipo]
        
        if u_tipo == 'T':
            fecha_ref_default = u_fecha
            target_sesion_default = "Noche"
        elif u_tipo == 'N':
            fecha_ref_default = u_fecha + timedelta(days=1)
            target_sesion_default = "Tarde"
            
        info_ultimo_sorteo = {'fecha': ultimo_registro['Fecha'], 'tipo': u_tipo}

    else:
        st.sidebar.warning("No hay datos.")

    # --- CONFIGURACION DE FECHA DE ANALISIS ---
    modo_sorteo = st.sidebar.radio("Análisis:", ["General", "Tarde", "Noche"])
    modo_fecha = st.sidebar.radio("Fecha Ref:", ["Auto (Último Dato)", "Personalizado"])
    
    fecha_ref = pd.to_datetime(fecha_ref_default)
    target_sesion = target_sesion_default
    
    if modo_fecha == "Personalizado":
        fecha_ref = st.sidebar.date_input("Fecha:", datetime.now().date())
        fecha_ref = pd.to_datetime(fecha_ref)
        sesion_estado = st.sidebar.radio("Estado:", ["Antes de Tarde", "Después de Tarde"], horizontal=False)
        if sesion_estado == "Antes de Tarde": target_sesion = "Tarde"
        else: target_sesion = "Noche"

    if st.sidebar.button("🔄 Recargar"): 
        st.cache_data.clear()
        st.rerun()
    
    if modo_sorteo == "Tarde": df_analisis = df_full[df_full['Tipo_Sorteo'] == 'T'].copy()
    elif modo_sorteo == "Noche": df_analisis = df_full[df_full['Tipo_Sorteo'] == 'N'].copy()
    else: df_analisis = df_full.copy()
    
    if df_analisis.empty: st.warning("Sin datos."); st.stop()

    orden_sesiones = {'Tarde': 0, 'Noche': 1}
    target_val = orden_sesiones[target_sesion]
    
    if target_val == 0: 
        df_backtest = df_analisis[df_analisis['Fecha'] < fecha_ref].copy()
    else: 
        df_backtest = df_analisis[(df_analisis['Fecha'] < fecha_ref) | ((df_analisis['Fecha'] == fecha_ref) & (df_analisis['Tipo_Sorteo'] == 'T'))].copy()

    # --- HISTORIAL DE COMBINACIONES ---
    st.header("📜 Historial de Combinaciones y Estados")
    
    df_historial_perfiles_full = obtener_historial_perfiles_cacheado(df_full, RUTA_CACHE)
    
    if not df_historial_perfiles_full.empty:
        df_hist_view = df_historial_perfiles_full.copy()
        df_hist_view['Decena'] = df_hist_view['Numero'] // 10
        df_hist_view['Unidad'] = df_hist_view['Numero'] % 10
        
        sort_map = {'Noche': 1, 'Tarde': 0}
        df_hist_view['sort_key'] = df_hist_view['Sorteo'].map(sort_map)
        df_hist_view = df_hist_view.sort_values(by=['Fecha', 'sort_key'], ascending=[False, False])
        
        df_hist_view['Fecha'] = df_hist_view['Fecha'].dt.strftime('%d/%m/%Y')
        df_hist_view = df_hist_view.rename(columns={'Perfil': 'Estado Salida'})
        
        cols_display = ['Fecha', 'Sorteo', 'Numero', 'Decena', 'Unidad', 'Estado Salida']
        st.dataframe(df_hist_view[cols_display].head(30), hide_index=True, use_container_width=True)
    else:
        st.warning("No se pudo generar el historial de perfiles.")

    st.markdown("---")

    # 1. Estado Actual
    df_oport_dec, df_oport_uni = analizar_oportunidad_por_digito(df_backtest, fecha_ref)
    
    st.header(f"🎯 Estado de Dígitos ({target_sesion} {fecha_ref.strftime('%d/%m')})")
    
    if info_ultimo_sorteo:
        tipo_nombre = {'T': 'Tarde', 'N': 'Noche'}[info_ultimo_sorteo['tipo']]
        fecha_txt = info_ultimo_sorteo['fecha'].strftime('%d/%m/%Y')
        st.caption(f"✅ Cálculo basado en datos hasta: **{tipo_nombre} {fecha_txt}**.")
    
    col1, col2 = st.columns(2)
    with col1: st.dataframe(df_oport_dec.sort_values('Punt. Base', ascending=False), hide_index=True)
    with col2: st.dataframe(df_oport_uni.sort_values('Punt. Base', ascending=False), hide_index=True)
        
    # 2. Análisis de Perfiles
    st.markdown("---")
    st.header("📅 Análisis de Perfiles (Motor Mejorado)")
    
    if st.button("🚀 Ejecutar Análisis", type="primary"):
        with st.spinner("Analizando..."):
            if not df_historial_perfiles_full.empty:
                if target_val == 0: 
                     df_historial_perfiles = df_historial_perfiles_full[df_historial_perfiles_full['Fecha'] < fecha_ref].copy()
                else: 
                     df_historial_perfiles = df_historial_perfiles_full[(df_historial_perfiles_full['Fecha'] < fecha_ref) | ((df_historial_perfiles_full['Fecha'] == fecha_ref) & (df_historial_perfiles_full['Sorteo'] == 'Tarde'))].copy()

                if not df_historial_perfiles.empty:
                    df_stats, transizioni, ultimo_perfil = analizar_estadisticas_perfiles(df_historial_perfiles, fecha_ref)
                    estabilidad_digitos = calcular_estabilidad_historica_digitos(df_backtest)
                    generar_sugerencia_fusionada(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos)
                    
                    st.markdown("---")
                    st.subheader("📊 Estadística de Perfiles (Completa)")
                    cols_tabla = ['Perfil', 'Frecuencia', 'Veces Normal', 'Veces Vencido', 'Veces Muy Vencido', 'Estado Actual', 'Estabilidad', 'Tiempo Limite', 'Alerta']
                    df_display = df_stats[cols_tabla].copy()
                    st.dataframe(df_display.sort_values('Frecuencia', ascending=False), hide_index=True)

    # 3. BACKTEST
    st.markdown("---")
    st.header("🧪 Backtesting")
    dias_back = st.slider("Días a simular", 7, 60, 30, key="slider_backtest")
    
    if st.button("▶️ Iniciar Backtest"):
        df_res, aciertos, total = ejecutar_backtest(df_full, dias_back)
        
        st.subheader("📊 Resultados")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sorteos", total)
        col2.metric("Aciertos", aciertos)
        col3.metric("Efectividad", f"{round((aciertos/total)*100, 1) if total > 0 else 0} %")
        
        with st.expander("Ver detalle"):
            st.dataframe(df_res, hide_index=True)

if __name__ == "__main__":
    main()