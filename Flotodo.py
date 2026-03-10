# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import os
import time 
from collections import defaultdict, Counter
import unicodedata
import calendar
from dateutil.relativedelta import relativedelta

# --- CONFIGURACIÓN DE LA RUTA ---
RUTA_CSV = 'Flotodo.csv'
RUTA_CACHE = 'cache_perfiles_florida.csv'

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Florida - Análisis de Sorteos",
    page_icon="🌴",
    layout="wide"
)

st.title("🌴 Florida - Análisis de Sorteos")
st.markdown("Sistema de Análisis con Lógica de Estabilidad Corregida y Backtesting.")

# --- FUNCIONES AUXILIARES Y DE CARGA ---

def remove_accents(input_str):
    if not isinstance(input_str, str): return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

@st.cache_resource
def cargar_datos_flotodo(_ruta_csv):
    try:
        if not os.path.exists(_ruta_csv):
            st.error(f"❌ Error: No se encontró el archivo en: {_ruta_csv}")
            st.stop()
        
        with open(_ruta_csv, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        
        if not lines: st.error("Archivo vacío."); st.stop()
        
        header_line = lines[0].strip()
        column_names = header_line.split(';')
        data = []
        for line in lines[1:]:
            if line.strip():
                values = line.strip().split(';')
                if len(values) >= 6: 
                    data.append(values)
        
        df = pd.DataFrame(data, columns=column_names[:6]) 
        
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
        
        df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper().map({
            'TARDE': 'T', 'T': 'T', 'NOCHE': 'N', 'N': 'N'
        }).fillna('OTRO')
        df = df[df['Tipo_Sorteo'].isin(['T', 'N'])]
        
        df_procesado = []
        for _, row in df.iterrows():
            try:
                fecha = row['Fecha']
                tipo = row['Tipo_Sorteo']
                fijo = int(row['Fijo']) if pd.notna(row.get('Fijo', 0)) else 0
                p1 = int(row['Primer_Corrido']) if pd.notna(row.get('Primer_Corrido', 0)) else 0
                p2 = int(row['Segundo_Corrido']) if pd.notna(row.get('Segundo_Corrido', 0)) else 0
                centena = int(row['Centena']) if pd.notna(row.get('Centena', 0)) else 0
                
                df_procesado.append({'Fecha': fecha, 'Tipo_Sorteo': tipo, 'Numero': centena, 'Posicion': 'Centena'})
                df_procesado.append({'Fecha': fecha, 'Tipo_Sorteo': tipo, 'Numero': fijo, 'Posicion': 'Fijo'})
                df_procesado.append({'Fecha': fecha, 'Tipo_Sorteo': tipo, 'Numero': p1, 'Posicion': '1er Corrido'})
                df_procesado.append({'Fecha': fecha, 'Tipo_Sorteo': tipo, 'Numero': p2, 'Posicion': '2do Corrido'})
            except: continue
        
        df_historial = pd.DataFrame(df_procesado)
        df_historial['Numero'] = pd.to_numeric(df_historial['Numero'], errors='coerce')
        df_historial.dropna(subset=['Numero'], inplace=True)
        df_historial['Numero'] = df_historial['Numero'].astype(int)
        
        draw_order_map = {'T': 0, 'N': 1}
        df_historial['draw_order'] = df_historial['Tipo_Sorteo'].map(draw_order_map)
        df_historial['sort_key'] = df_historial['Fecha'] + pd.to_timedelta(df_historial['draw_order'], unit='h')
        df_historial = df_historial.sort_values(by='sort_key').reset_index(drop=True)
        df_historial.drop(columns=['draw_order', 'sort_key'], inplace=True)
        
        return df_historial
    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        st.stop()

def calcular_estado_actual(gap, promedio_gap):
    if pd.isna(promedio_gap) or promedio_gap == 0: return "Normal"
    if gap <= promedio_gap: return "Normal"
    elif gap > (promedio_gap * 1.5): return "Muy Vencido"
    else: return "Vencido"

def obtener_df_temperatura(contador):
    df = pd.DataFrame.from_dict(contador, orient='index', columns=['Frecuencia'])
    df = df.reset_index().rename(columns={'index': 'Dígito'})
    df = df.sort_values('Frecuencia', ascending=False).reset_index(drop=True)
    df['Temperatura'] = '🧊 Frío'
    if len(df) >= 3: df.loc[0:2, 'Temperatura'] = '🔥 Caliente'
    if len(df) >= 7: df.loc[6:9, 'Temperatura'] = '🧊 Frío'
    if len(df) >= 3: df.loc[3:5, 'Temperatura'] = '🟡 Tibio'
    return df

# --- FUNCIONES DE ANÁLISIS ---

def get_full_state_dataframe(df_historial, fecha_referencia):
    df_fijos_hist = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    df_fijos_filtrado = df_fijos_hist[df_fijos_hist['Fecha'] < fecha_referencia].copy()
    if df_fijos_filtrado.empty: return pd.DataFrame(), {}
    
    df_maestro = pd.DataFrame({'Numero': range(100)})
    primera_fecha_historica = df_fijos_hist['Fecha'].min()
    historicos_numero = {}
    for i in range(100):
        fechas_i = df_fijos_filtrado[df_fijos_filtrado['Numero'] == i]['Fecha'].sort_values()
        gaps = fechas_i.diff().dt.days.dropna()
        historicos_numero[i] = gaps.median() if len(gaps) > 0 else (fecha_referencia - primeira_fecha_historica).days

    df_maestro['Decena'] = df_maestro['Numero'] // 10
    df_maestro['Unidad'] = df_maestro['Numero'] % 10
    ultima_aparicion = df_fijos_filtrado.groupby('Numero')['Fecha'].max().reindex(range(100)).fillna(primera_fecha_historica)
    gap_num = (fecha_referencia - ultima_aparicion).dt.days
    df_maestro['Salto_Numero'] = gap_num
    df_maestro['Estado_Numero'] = df_maestro.apply(lambda row: calcular_estado_actual(row['Salto_Numero'], historicos_numero[row['Numero']]), axis=1)
    df_maestro['Total_Salidas_Historico'] = df_fijos_filtrado['Numero'].value_counts().reindex(range(100)).fillna(0)
    return df_maestro, historicos_numero

def analizar_oportunidad_por_digito(df_historial, fecha_referencia):
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
        
        p_proact_d = min(49, (gap_d / prom_d * 50)) if prom_d > 0 and ed == 'Normal' else 0
        p_proact_u = min(49, (gap_u / prom_u * 50)) if prom_u > 0 and eu == 'Normal' else 0
        
        p_temp_map = {'🔥 Caliente': 30, '🟡 Tibio': 20, '🧊 Frío': 10}
        p_temp_d = p_temp_map.get(mapa_temp_dec.get(i, '🟡 Tibio'), 20)
        p_temp_u = p_temp_map.get(mapa_temp_uni.get(i, '🟡 Tibio'), 20)
        
        res_dec.append({
            'Dígito': i, 'Temperatura': mapa_temp_dec.get(i, '🟡 Tibio'), 'Estado': ed, 
            'Punt. Base': p_base_d, 'Punt. Temp': p_temp_d, 'Puntuación': p_base_d + p_proact_d + p_temp_d
        })
        res_uni.append({
            'Dígito': i, 'Temperatura': mapa_temp_uni.get(i, '🟡 Tibio'), 'Estado': eu, 
            'Punt. Base': p_base_u, 'Punt. Temp': p_temp_u, 'Puntuación': p_base_u + p_proact_u + p_temp_u
        })

    return pd.DataFrame(res_dec), pd.DataFrame(res_uni)

# --- GESTOR DE CACHE ---
def obtener_historial_perfiles_cacheado(df_full, ruta_cache):
    df_fijos = df_full[df_full['Posicion'] == 'Fijo'].copy()
    df_cache = pd.DataFrame()
    if os.path.exists(ruta_cache):
        try:
            df_cache = pd.read_csv(ruta_cache, parse_dates=['Fecha'])
        except:
            df_cache = pd.DataFrame() 

    if not df_cache.empty:
        ultima_fecha_cache = df_cache['Fecha'].max()
        df_nuevos = df_fijos[df_fijos['Fecha'] > ultima_fecha_cache].copy()
        if df_nuevos.empty:
            return df_cache 
    else:
        df_nuevos = df_fijos 

    if df_nuevos.empty:
        return df_cache

    if not df_cache.empty:
        hist_decenas = defaultdict(list)
        hist_unidades = defaultdict(list)
        for _, row in df_cache.iterrows():
            num = int(row['Numero'])
            fecha = row['Fecha']
            hist_decenas[num // 10].append(fecha)
            hist_unidades[num % 10].append(fecha)
    else:
        hist_decenas = defaultdict(list)
        hist_unidades = defaultdict(list)
        
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
        
        nuevos_registros.append({
            'Fecha': fecha_actual,
            'Sorteo': 'Noche' if tipo_actual == 'N' else 'Tarde',
            'Numero': num_actual,
            'Perfil': perfil
        })
        
        hist_decenas[dec].append(fecha_actual)
        hist_unidades[uni].append(fecha_actual)
    
    if nuevos_registros:
        df_nuevos_cache = pd.DataFrame(nuevos_registros)
        if not df_cache.empty:
            df_final = pd.concat([df_cache, df_nuevos_cache], ignore_index=True)
        else:
            df_final = df_nuevos_cache
            
        df_final.to_csv(ruta_cache, index=False)
        return df_final
    else:
        return df_cache

# --- ANALISIS DE ESTADISTICAS AVANZADAS (MODIFICADO) ---
def analizar_estadisticas_perfiles(df_historial_perfiles, fecha_referencia):
    historial_fechas_perfiles = defaultdict(list)
    transiciones = Counter()
    ultimo_perfil = None
    
    for _, row in df_historial_perfiles.iterrows():
        perfil = row['Perfil']
        fecha = row['Fecha']
        historial_fechas_perfiles[perfil].append(fecha)
        
        if ultimo_perfil:
            transiciones[(ultimo_perfil, perfil)] += 1
        ultimo_perfil = perfil
        
    analisis_perfiles = []
    
    for perfil, fechas in historial_fechas_perfiles.items():
        fechas_ordenadas = sorted(fechas)
        ultima_fecha = fechas_ordenadas[-1]
        
        gaps = []
        for k in range(1, len(fechas_ordenadas)):
            gaps.append((fechas_ordenadas[k] - fechas_ordenadas[k-1]).days)
            
        mediana_gap = np.median(gaps) if gaps else 0
        gap_actual = (fecha_referencia - ultima_fecha).days
        estado_actual = calcular_estado_actual(gap_actual, mediana_gap)
        
        estados_historicos = []
        if gaps:
            for g in gaps:
                estados_historicos.append(calcular_estado_actual(g, mediana_gap))
        
        total_hist = len(estados_historicos)
        muy_vencidos_count = estados_historicos.count('Muy Vencido')
        estabilidad_pct = ((total_hist - muy_vencidos_count) / total_hist * 100) if total_hist > 0 else 0
        
        alerta_recuperacion = False
        if estabilidad_pct > 60 and estado_actual in ['Vencido', 'Muy Vencido']:
            alerta_recuperacion = True
            
        veces_anterior = transiciones.get((perfil, perfil), 0)
        prob_repeticion = (veces_anterior / total_hist * 100) if total_hist > 0 else 0
        
        semanas_con_presencia = set([f.isocalendar()[1] for f in fechas_ordenadas])
        estabilidad_semanal = len(semanas_con_presencia)
        
        # --- NUEVO: CALCULO DE TIEMPO LÍMITE ---
        limite_muy_vencido = mediana_gap * 1.5
        tiempo_limite_str = "-"
        
        if mediana_gap > 0:
            if estado_actual == "Muy Vencido":
                # Cuántos días lleva excedido
                exceso = int(gap_actual - limite_muy_vencido)
                tiempo_limite_str = f"🔴 +{exceso} días exceso"
            elif estado_actual == "Vencido":
                # Cuántos días faltan para ser Muy Vencido
                faltan = int(limite_muy_vencido - gap_actual)
                tiempo_limite_str = f"🟠 Faltan {faltan} días"
            else:
                # Cuántos días faltan para siquiera vencerse
                faltan_vencido = int(mediana_gap - gap_actual)
                tiempo_limite_str = f"🟢 Faltan {faltan_vencido} días"
        
        analisis_perfiles.append({
            'Perfil': perfil,
            'Frecuencia': total_hist + 1,
            'Última Fecha': ultima_fecha.strftime('%d/%m/%Y'),
            'Gap Actual': gap_actual,
            'Mediana Gap': round(mediana_gap, 1),
            'Estado Actual': estado_actual,
            'Estabilidad %': round(estabilidad_pct, 1),
            '⏳ Tiempo Límite': tiempo_limite_str, # Nueva columna
            'Alerta': '⚠️ RECUPERAR' if alerta_recuperacion else '-',
            'Prob. Repetición %': round(prob_repeticion, 1),
            'Semanas Activo': estabilidad_semanal
        })
        
    df_stats = pd.DataFrame(analisis_perfiles)
    return df_stats, transiciones, ultimo_perfil

# --- MOTORES DE PREDICCION ---

def obtener_prediccion_numeros_lista(df_stats, transiciones, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref):
    scores = []
    for _, row in df_stats.iterrows():
        p = row['Perfil']
        score = 0
        estado = row['Estado Actual']
        if row['Alerta'] == '⚠️ RECUPERAR': score += 150 
        else:
            if estado == 'Vencido': score += 70 
            elif estado == 'Normal': score += 50 
            elif estado == 'Muy Vencido': score += 30 
        score += row['Estabilidad %'] * 0.5
        trans_count = transiciones.get((ultimo_perfil, p), 0)
        score += trans_count * 10 
        scores.append({'Perfil': p, 'Score': score, 'Estado': estado, 'Frec': row['Frecuencia'], 'Alerta': row['Alerta']})
    
    df_scores = pd.DataFrame(scores).sort_values('Score', ascending=False)
    top_3 = df_scores.head(3)
    
    map_estado_dec = df_oport_dec.set_index('Dígito')['Estado'].to_dict()
    map_estado_uni = df_oport_uni.set_index('Dígito')['Estado'].to_dict()
    map_temp_dec = df_oport_dec.set_index('Dígito')['Temperatura'].to_dict()
    map_temp_uni = df_oport_uni.set_index('Dígito')['Temperatura'].to_dict()
    
    temp_val = {'🔥 Caliente': 3, '🟡 Tibio': 2, '🧊 Frío': 1}
    df_hist_nums = df_historial_perfiles.groupby('Numero')['Fecha'].max()
    candidatos_totales = []
    
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
                temp_d_score = temp_val.get(map_temp_dec.get(d, '🧊 Frío'), 1)
                temp_u_score = temp_val.get(map_temp_uni.get(u, '🧊 Frío'), 1)
                temp_total = temp_d_score + temp_u_score
                candidatos_totales.append({
                    'Numero': num, 'Perfil': perfil, 'Score': row['Score'], 
                    'Gap_Num': gap_n, 'Temp_Score': temp_total
                })
                
    df_cands = pd.DataFrame(candidatos_totales)
    if df_cands.empty: return []
    df_cands = df_cands.sort_values(['Score', 'Temp_Score'], ascending=[False, False]).drop_duplicates(subset=['Numero'])
    df_cands = df_cands.sort_values(by=['Score', 'Temp_Score', 'Gap_Num'], ascending=[False, False, False])
    return df_cands.head(30)['Numero'].tolist()

def generar_sugerencia_fusionada(df_stats, transiciones, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref):
    st.subheader("🤖 Sugerencia Inteligente Fusionada")
    
    # Desglose de Alertas
    st.markdown("### 🚨 Detalle de Alertas Activas")
    st.markdown("Si hay alertas, aquí verás los números exactos que las componen:")
    
    map_estado_dec = df_oport_dec.set_index('Dígito')['Estado'].to_dict()
    map_estado_uni = df_oport_uni.set_index('Dígito')['Estado'].to_dict()
    
    alertas_activas = df_stats[df_stats['Alerta'] == '⚠️ RECUPERAR']
    
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
                    nums_alerta.append(f"{d}{u}")
            
            st.markdown(f"**Perfil Alertado: `{perfil_name}`**")
            st.markdown(f"📍 **Estado:** {row_alert['Estado Actual']} | ⏳ **{row_alert['⏳ Tiempo Límite']}**")
            
            col_info_1, col_info_2 = st.columns(2)
            with col_info_1:
                st.caption(f"Decenas '{ed_req}': {decenas_cumplen}")
            with col_info_2:
                st.caption(f"Unidades '{eu_req}': {unidades_cumplen}")
            
            if nums_alerta:
                st.success(f"🔢 Números que forman esta alerta ({len(nums_alerta)}): {' - '.join(nums_alerta)}")
            else:
                st.warning("No se encontraron números que cumplan estrictamente con la combinación actual de dígitos.")
            st.markdown("---")
    else:
        st.info("No hay alertas de recuperación activas para esta fecha.")

    # Predicción General
    st.markdown("### 🎲 Top 30 Números Sugeridos")
    lista_nums = obtener_prediccion_numeros_lista(df_stats, transiciones, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref)
    
    scores = []
    for _, row in df_stats.iterrows():
        p = row['Perfil']
        score = 0
        estado = row['Estado Actual']
        if row['Alerta'] == '⚠️ RECUPERAR': score += 150
        else:
            if estado == 'Vencido': score += 70 
            elif estado == 'Normal': score += 50 
            elif estado == 'Muy Vencido': score += 30
        score += row['Estabilidad %'] * 0.5
        trans_count = transiciones.get((ultimo_perfil, p), 0)
        score += trans_count * 10 
        scores.append({'Perfil': p, 'Score': score, 'Estado': estado, 'Frec': row['Frecuencia'], 'Alerta': row['Alerta']})
    
    df_scores = pd.DataFrame(scores).sort_values('Score', ascending=False)
    top_3 = df_scores.head(3)

    st.markdown(f"**Último perfil sorteo:** `{ultimo_perfil}`")
    st.markdown("**Perfiles seleccionados:** (Prioridad Alerta > Transición > Frecuencia)")
    
    def highlight_alerts(s):
        return ['background-color: #FFD700' if v == '⚠️ RECUPERAR' else '' for v in s]
    
    st.dataframe(top_3[['Perfil', 'Score', 'Estado', 'Alerta', 'Frec']].style.apply(highlight_alerts, subset=['Alerta']), hide_index=True)
    
    # Mapas necesarios para visualización
    # Recuperamos Estado y Temperatura
    map_estado_dec = df_oport_dec.set_index('Dígito')['Estado'].to_dict()
    map_estado_uni = df_oport_uni.set_index('Dígito')['Estado'].to_dict()
    
    # Helper para colores de estado
    def get_state_color(state):
        if state == 'Normal': return '#00FF00' # Verde
        elif state == 'Vencido': return '#FFA500' # Naranja
        elif state == 'Muy Vencido': return '#FF0000' # Rojo
        return '#FFFFFF'
    
    if not lista_nums:
        st.warning("No se generaron candidatos.")
        return

    cols = st.columns(6)
    for idx, num in enumerate(lista_nums):
        num_str = f"{num:02d}"
        d_int = int(num_str[0])
        u_int = int(num_str[1])
        
        # Obtener Estados
        ed = map_estado_dec.get(d_int, "?")
        eu = map_estado_uni.get(u_int, "?")
        
        # Colores
        color_d = get_state_color(ed)
        color_u = get_state_color(eu)
        
        # HTML con composición de estado
        cols[idx % 6].markdown(f"""
        <div style="background-color:#1E1E1E; padding:10px; border-radius:5px; text-align:center; border: 2px solid #444;">
            <h3 style="margin:0; color:#00FF00;">{num:02d}</h3>
            <small style="color:{color_d}; font-weight:bold;">{ed}</small>
            <small style="color:white;"> - </small>
            <small style="color:{color_u}; font-weight:bold;">{eu}</small>
        </div>
        """, unsafe_allow_html=True)

# --- ALMANAQUE ---
def analizar_almanaque_combinaciones(df_historial, dia_inicio, dia_fin, meses_atras, fecha_referencia):
    df_fijos = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    fecha_hoy = fecha_referencia
    
    perfiles_contador = Counter()
    numeros_por_perfil = defaultdict(Counter)
    historico_estados_digitos = [] 
    
    todos_nums_bloques = []
    
    for i in range(1, meses_atras + 1):
        f_obj = fecha_hoy - relativedelta(months=i)
        try:
            last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
            f_inicio = datetime(f_obj.year, f_obj.month, min(dia_inicio, last_day))
            f_fin = datetime(f_obj.year, f_obj.month, min(dia_fin, last_day))
            if f_inicio > f_fin: continue
            
            df_hist_antes = df_fijos[df_fijos['Fecha'] < f_inicio].copy()
            estados_decenas = {}
            estados_unidades = {}
            
            for d in range(10):
                fechas_d = df_hist_antes[df_hist_antes['Numero'] // 10 == d]['Fecha'].sort_values()
                if not fechas_d.empty:
                    gaps = fechas_d.diff().dt.days.dropna()
                    prom = gaps.median() if len(gaps) > 0 else 0
                    gap = (f_inicio - fechas_d.max()).days
                    estados_decenas[d] = calcular_estado_actual(gap, prom)
                else: estados_decenas[d] = 'Normal'
                
                fechas_u = df_hist_antes[df_hist_antes['Numero'] % 10 == d]['Fecha'].sort_values()
                if not fechas_u.empty:
                    gaps = fechas_u.diff().dt.days.dropna()
                    prom = gaps.median() if len(gaps) > 0 else 0
                    gap = (f_inicio - fechas_u.max()).days
                    estados_unidades[d] = calcular_estado_actual(gap, prom)
                else: estados_unidades[d] = 'Normal'
            
            historico_estados_digitos.append({
                'Mes': f_obj.strftime("%B %Y"), 'Fecha Inicio': f_inicio,
                'Estados Decenas': estados_decenas, 'Estados Unidades': estados_unidades
            })
            
            df_bloque = df_fijos[(df_fijos['Fecha'] >= f_inicio) & (df_fijos['Fecha'] <= f_fin)].copy()
            if df_bloque.empty: continue
            
            todos_nums_bloques.extend(df_bloque['Numero'].tolist())
            
            for _, row in df_bloque.iterrows():
                num = row['Numero']
                dec = num // 10
                uni = num % 10
                estado_dec = estados_decenas.get(dec, 'Normal')
                estado_uni = estados_unidades.get(uni, 'Normal')
                combinacion = f"{estado_dec}-{estado_uni}"
                
                perfiles_contador[combinacion] += 1
                numeros_por_perfil[combinacion][num] += 1
        except: pass
    
    df_tendencias = pd.DataFrame.from_dict(perfiles_contador, orient='index', columns=['Frecuencia']).reset_index()
    df_tendencias.columns = ['Combinación', 'Frecuencia']
    df_tendencias = df_tendencias.sort_values('Frecuencia', ascending=False)
    
    contador_decenas_hist = Counter([n // 10 for n in todos_nums_bloques])
    contador_unidades_hist = Counter([n % 10 for n in todos_nums_bloques])
    df_temp_dec_hist = obtener_df_temperatura(contador_decenas_hist)
    df_temp_uni_hist = obtener_df_temperatura(contador_unidades_hist)
            
    return df_tendencias, numeros_por_perfil, historico_estados_digitos, df_temp_dec_hist, df_temp_uni_hist

# --- BACKTEST ---
def ejecutar_backtest(df_full, dias_atras):
    hoy = datetime.now().date()
    fecha_inicio = hoy - timedelta(days=dias_atras)
    
    resultados = []
    aciertos = 0
    total_sorteos = 0
    
    df_cache_full = obtener_historial_perfiles_cacheado(df_full, RUTA_CACHE)
    
    st.info(f"Simulando {dias_atras} días hacia atrás...")
    progress_bar = st.progress(0)
    
    for i in range(dias_atras):
        current_date = hoy - timedelta(days=dias_atras - i - 1)
        fecha_ref = pd.to_datetime(current_date)
        
        for sorteo_tipo in ['T', 'N']:
            if sorteo_tipo == 'T':
                mask_datos = (df_full['Fecha'] < fecha_ref)
            else:
                mask_datos = (df_full['Fecha'] < fecha_ref) | \
                             ((df_full['Fecha'] == fecha_ref) & (df_full['Tipo_Sorteo'] == 'T'))
            
            df_disponible = df_full[mask_datos].copy()
            mask_real = (df_full['Fecha'] == fecha_ref) & (df_full['Tipo_Sorteo'] == sorteo_tipo) & (df_full['Posicion'] == 'Fijo')
            
            if mask_real.sum() == 0: continue
                
            resultado_real = df_full[mask_real]['Numero'].iloc[0]
            total_sorteos += 1
            
            if sorteo_tipo == 'T':
                mask_cache = (df_cache_full['Fecha'] < fecha_ref)
            else:
                mask_cache = (df_cache_full['Fecha'] < fecha_ref) | \
                             ((df_cache_full['Fecha'] == fecha_ref) & (df_cache_full['Sorteo'] == 'Tarde'))
            
            df_cache_sim = df_cache_full[mask_cache].copy()
            
            if df_disponible.empty or df_cache_sim.empty: continue
            
            df_oport_dec, df_oport_uni = analizar_oportunidad_por_digito(df_disponible, fecha_ref)
            df_stats, transiciones, ultimo_perfil = analizar_estadisticas_perfiles(df_cache_sim, fecha_ref)
            prediccion = obtener_prediccion_numeros_lista(df_stats, transiciones, ultimo_perfil, df_oport_dec, df_oport_uni, df_cache_sim, fecha_ref)
            
            hit = resultado_real in prediccion
            if hit: aciertos += 1
            
            resultados.append({
                'Fecha': current_date, 'Sorteo': 'Tarde' if sorteo_tipo == 'T' else 'Noche',
                'Real': resultado_real, 'Acierto': '✅ SÍ' if hit else '❌ NO'
            })
        
        progress_bar.progress((i + 1) / dias_atras)
        
    progress_bar.empty()
    return pd.DataFrame(resultados), aciertos, total_sorteos

# --- MAIN ---
def main():
    st.sidebar.header("⚙️ Opciones")
    
    with st.sidebar.expander("📝 Agregar Sorteo", False):
        f_nueva = st.date_input("Fecha", datetime.now().date())
        ses = st.radio("Sesión", ["Tarde", "Noche"], horizontal=True)
        cent = st.number_input("Centena", 0, 999, 0, key="inp_cent")
        fij = st.number_input("Fijo", 0, 99, 0, key="inp_fijo")
        c1 = st.number_input("1er Corrido", 0, 99, 0, key="inp_c1")
        c2 = st.number_input("2do Corrido", 0, 99, 0, key="inp_c2")
        
        if st.button("💾 Guardar Sorteo"):
            s_code = "T" if "Tarde" in ses else "N"
            line = f"{f_nueva.strftime('%d/%m/%Y')};{s_code};{cent};{fij};{c1};{c2}\n"
            try:
                with open(RUTA_CSV, 'a', encoding='latin-1') as file: file.write(line)
                st.success("¡Guardado con éxito!")
                time.sleep(1)
                st.cache_resource.clear()
                st.rerun()
            except Exception as err: st.error(f"Error al guardar: {err}")

    modo_sorteo = st.sidebar.radio("Análisis:", ["General", "Tarde", "Noche"])
    modo_fecha = st.sidebar.radio("Fecha Ref:", ["Hoy", "Personalizado"])
    
    fecha_ref = pd.Timestamp.now(tz=None).normalize()
    target_sesion = "Tarde"
    
    if modo_fecha == "Personalizado":
        fecha_ref = st.sidebar.date_input("Fecha:", datetime.now().date())
        fecha_ref = pd.to_datetime(fecha_ref)
        sesion_estado = st.sidebar.radio("Estado del Día:", ["Antes de Tarde", "Después de Tarde"], horizontal=False)
        
        if sesion_estado == "Antes de Tarde":
            target_sesion = "Tarde"
        else:
            target_sesion = "Noche"
        st.sidebar.caption(f"Predicción para: {target_sesion}")
    else:
        now = datetime.now()
        if now.hour < 14 or (now.hour == 14 and now.minute < 30):
            target_sesion = "Tarde"
        elif now.hour < 19 or (now.hour == 19 and now.minute < 30):
            target_sesion = "Noche"
        else:
            target_sesion = "Tarde"
            fecha_ref = fecha_ref + timedelta(days=1)
            st.sidebar.info(f"Auto: Próxima Tarde ({fecha_ref.strftime('%d/%m')})")

    if st.sidebar.button("🔄 Recargar"): st.rerun()

    df_full = cargar_datos_flotodo(RUTA_CSV)
    
    if "Tarde" in modo_sorteo: df_analisis = df_full[df_full['Tipo_Sorteo'] == 'T'].copy()
    elif "Noche" in modo_sorteo: df_analisis = df_full[df_full['Tipo_Sorteo'] == 'N'].copy()
    else: df_analisis = df_full.copy()
    
    if df_analisis.empty: st.warning("Sin datos."); st.stop()

    if modo_fecha == "Personalizado":
        if target_sesion == "Tarde":
            df_backtest = df_analisis[df_analisis['Fecha'] < fecha_ref].copy()
        else:
            mask = (df_analisis['Fecha'] < fecha_ref) | \
                   ((df_analisis['Fecha'] == fecha_ref) & (df_analisis['Tipo_Sorteo'] == 'T'))
            df_backtest = df_analisis[mask].copy()
    else:
        if target_sesion == "Tarde" and fecha_ref > pd.Timestamp.now(tz=None).normalize():
             df_backtest = df_analisis[df_analisis['Fecha'] < fecha_ref].copy()
        elif target_sesion == "Tarde":
             df_backtest = df_analisis[df_analisis['Fecha'] < fecha_ref].copy()
        else:
             mask = (df_analisis['Fecha'] < fecha_ref) | \
                    ((df_analisis['Fecha'] == fecha_ref) & (df_analisis['Tipo_Sorteo'] == 'T'))
             df_backtest = df_analisis[mask].copy()

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Últimos Sorteos")
    df_info = df_full[df_full['Posicion'].isin(['Fijo', 'Centena', '1er Corrido', '2do Corrido'])]
    def get_num(df, pos):
        val = df[df['Posicion'] == pos]['Numero']
        return f"{int(val.iloc[0]):02d}" if not val.empty else "-"
    
    ultima_noche = df_info[(df_info['Tipo_Sorteo'] == 'N') & (df_info['Posicion'] == 'Fijo')].tail(1)
    if not ultima_noche.empty:
        fecha_n = ultima_noche['Fecha'].iloc[0]
        datos_n = df_info[(df_info['Fecha'] == fecha_n) & (df_info['Tipo_Sorteo'] == 'N')]
        st.sidebar.markdown(f"**🌙 Noche ({fecha_n.strftime('%d/%m')})**")
        st.sidebar.markdown(f"Fijo: {get_num(datos_n, 'Fijo')} | C: {get_num(datos_n, 'Centena')}")

    ultima_tarde = df_info[(df_info['Tipo_Sorteo'] == 'T') & (df_info['Posicion'] == 'Fijo')].tail(1)
    if not ultima_tarde.empty:
        fecha_t = ultima_tarde['Fecha'].iloc[0]
        dados_t = df_info[(df_info['Fecha'] == fecha_t) & (df_info['Tipo_Sorteo'] == 'T')]
        st.sidebar.markdown(f"**🌞 Tarde ({fecha_t.strftime('%d/%m')})**")
        st.sidebar.markdown(f"Fijo: {get_num(dados_t, 'Fijo')} | C: {get_num(dados_t, 'Centena')}")

    # 1. Estado Actual
    df_estados_num, hist_num = get_full_state_dataframe(df_backtest, fecha_ref)
    df_oport_dec, df_oport_uni = analizar_oportunidad_por_digito(df_backtest, fecha_ref)
    
    st.header(f"🎯 Estado de Dígitos (Objetivo: {target_sesion} {fecha_ref.strftime('%d/%m')})")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Decenas")
        st.dataframe(df_oport_dec.sort_values('Puntuación', ascending=False), hide_index=True)
    with col2:
        st.subheader("Unidades")
        st.dataframe(df_oport_uni.sort_values('Puntuación', ascending=False), hide_index=True)
        
    # 2. Análisis de Perfiles
    st.markdown("---")
    st.header("📅 Análisis de Perfiles y Aprendizaje")
    
    if st.button("🚀 Ejecutar Análisis Inteligente", type="primary"):
        with st.spinner("Analizando estabilidad y patrones..."):
            df_historial_perfiles_full = obtener_historial_perfiles_cacheado(df_backtest, RUTA_CACHE)
            
            if modo_fecha == "Personalizado":
                if target_sesion == "Tarde":
                    df_historial_perfiles = df_historial_perfiles_full[df_historial_perfiles_full['Fecha'] < fecha_ref].copy()
                else:
                     mask = (df_historial_perfiles_full['Fecha'] < fecha_ref) | \
                            ((df_historial_perfiles_full['Fecha'] == fecha_ref) & (df_historial_perfiles_full['Sorteo'] == 'Tarde'))
                     df_historial_perfiles = df_historial_perfiles_full[mask].copy()
            else:
                df_historial_perfiles = df_historial_perfiles_full.copy()

            if df_historial_perfiles.empty:
                st.error("No hay datos históricos anteriores.")
            else:
                df_stats, transiciones, ultimo_perfil = analizar_estadisticas_perfiles(df_historial_perfiles, fecha_ref)
                generar_sugerencia_fusionada(df_stats, transiciones, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref)
                
                st.markdown("---")
                st.subheader("📊 Estado y Estabilidad de Perfiles")
                # Mostramos la nueva columna ⏳ Tiempo Límite
                st.dataframe(df_stats.sort_values('Frecuencia', ascending=False), hide_index=True, use_container_width=True)
                
                st.subheader("📜 Historial Reciente")
                df_reciente = df_historial_perfiles.sort_values(by=['Fecha', 'Sorteo'], ascending=[False, False]).head(20)
                df_reciente['Fecha'] = df_reciente['Fecha'].dt.strftime('%d/%m/%Y')
                st.dataframe(df_reciente, hide_index=True)

    # 3. Almanaque
    st.markdown("---")
    st.header("📆 Almanaque de Patrones (Rangos)")
    
    c_al1, c_al2, c_al3 = st.columns(3)
    with c_al1:
        al_d_ini = st.number_input("Día Inicio", 1, 31, 1, key="al_i")
        al_d_fin = st.number_input("Día Fin", 1, 31, 15, key="al_f")
    with c_al2:
        al_meses = st.slider("Meses Atrás", 1, 12, 3, key="al_m")
    with c_al3:
        st.markdown("### ")
        btn_al = st.button("🔮 Analizar Almanaque")
        
    if btn_al:
        with st.spinner("Calculando..."):
            df_tend, nums_perf, hist_dig, df_temp_d, df_temp_u = analizar_almanaque_combinaciones(
                df_backtest, al_d_ini, al_d_fin, al_meses, fecha_ref
            )
        
        st.subheader("Tendencias en Bloques")
        st.dataframe(df_tend, hide_index=True)
        
        filas_hist = []
        for bloque in hist_dig:
            ed = bloque['Estados Decenas']
            eu = bloque['Estados Unidades']
            dec_str = ", ".join([f"{k}:{v}" for k,v in sorted(ed.items())])
            uni_str = ", ".join([f"{k}:{v}" for k,v in sorted(eu.items())])
            filas_hist.append({
                'Mes': bloque['Mes'], 'Periodo': f"Días {al_d_ini}-{al_d_fin}",
                'Estado Decenas': dec_str, 'Estado Unidades': uni_str
            })
        
        if filas_hist:
            st.dataframe(pd.DataFrame(filas_hist), use_container_width=True, hide_index=True)

    # 4. BACKTEST
    st.markdown("---")
    st.header("🧪 Backtesting Automático")
    st.markdown("Simula cómo hubiese funcionado la predicción en días anteriores usando la lógica actual.")
    
    dias_back = st.slider("Días a simular hacia atrás", 7, 60, 30, key="slider_backtest")
    
    if st.button("▶️ Iniciar Backtest"):
        df_res, aciertos, total = ejecutar_backtest(df_full, dias_back)
        
        st.subheader("📊 Resultados del Backtest")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sorteos", total)
        col2.metric("Aciertos", aciertos)
        col3.metric("Tasa de Éxito", f"{round((aciertos/total)*100, 1) if total > 0 else 0} %")
        
        with st.expander("Ver detalle de sorteos"):
            st.dataframe(df_res, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()