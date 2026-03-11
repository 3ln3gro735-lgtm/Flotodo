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
st.markdown("Motor de Predicción Mejorado: Integrando Lógica de Corrección ('Doble Fallo') y Estabilidad.")

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
        
        # Ordenamiento cronológico estricto
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

# --- FUNCIONES DE ANALISIS ---

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
        historicos_numero[i] = gaps.median() if len(gaps) > 0 else (fecha_referencia - primera_fecha_historica).days

    df_maestro['Decena'] = df_maestro['Numero'] // 10
    df_maestro['Unidad'] = df_maestro['Numero'] % 10
    ultima_aparicion = df_fijos_filtrado.groupby('Numero')['Fecha'].max().reindex(range(100)).fillna(primera_fecha_historica)
    gap_num = (fecha_referencia - ultima_aparicion).dt.days
    df_maestro['Salto_Numero'] = gap_num
    df_maestro['Estado_Numero'] = df_maestro.apply(lambda row: calcular_estado_actual(row['Salto_Numero'], historicos_numero[row['Numero']]), axis=1)
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
        
        res_dec.append({
            'Dígito': i, 'Temperatura': mapa_temp_dec.get(i, '🟡 Tibio'), 'Estado': ed, 
            'Punt. Base': p_base_d
        })
        res_uni.append({
            'Dígito': i, 'Temperatura': mapa_temp_uni.get(i, '🟡 Tibio'), 'Estado': eu, 
            'Punt. Base': p_base_u
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

    # Identificador único para cada sorteo
    df_fijos['ID_Sorteo'] = df_fijos['Fecha'].astype(str) + "_" + df_fijos['Tipo_Sorteo']
    
    ids_en_cache = set()
    if not df_cache.empty:
        df_cache['ID_Sorteo'] = df_cache['Fecha'].astype(str) + "_" + df_cache['Sorteo'].map({'Noche': 'N', 'Tarde': 'T'})
        ids_en_cache = set(df_cache['ID_Sorteo'])
    
    # Filtrar sorteos nuevos
    df_nuevos = df_fijos[~df_fijos['ID_Sorteo'].isin(ids_en_cache)].copy()
    
    if df_nuevos.empty:
        if 'ID_Sorteo' in df_cache.columns: df_cache.drop(columns=['ID_Sorteo'], inplace=True)
        return df_cache
    
    df_nuevos = df_nuevos.sort_values(by=['Fecha', 'Tipo_Sorteo'])

    hist_decenas = defaultdict(list)
    hist_unidades = defaultdict(list)
    
    if not df_cache.empty:
        df_cache_sorted = df_cache.sort_values(by=['Fecha', 'Sorteo'], ascending=[True, True])
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
            df_final = pd.concat([df_cache.drop(columns=['ID_Sorteo'], errors='ignore'), df_nuevos_cache], ignore_index=True)
        else:
            df_final = df_nuevos_cache
            
        df_final.to_csv(ruta_cache, index=False)
        return df_final
    else:
        if 'ID_Sorteo' in df_cache.columns: df_cache.drop(columns=['ID_Sorteo'], inplace=True)
        return df_cache

def calcular_estabilidad_historica_digitos(df_full):
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

# --- ANALISIS ESTADISTICAS PERFILES ---
def analizar_estadisticas_perfiles(df_historial_perfiles, fecha_referencia):
    historial_fechas_perfiles = defaultdict(list)
    ultimo_suceso_perfil = {}
    transiciones = Counter()
    ultimo_perfil_global = None
    
    sort_map = {'Tarde': 0, 'Noche': 1}
    df_historial_perfiles = df_historial_perfiles.copy()
    df_historial_perfiles['sort_val'] = df_historial_perfiles['Sorteo'].map(sort_map)
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
        estado_actual = calcular_estado_actual(gap_actual, mediana_gap_actual)
        
        estados_historicos = [calcular_estado_actual(g, mediana_gap_actual) for g in gaps] if gaps else []
        total_hist = len(estados_historicos)
        muy_vencidos_count = estados_historicos.count('Muy Vencido')
        estabilidad_actual = ((total_hist - muy_vencidos_count) / total_hist * 100) if total_hist > 0 else 0
        
        alerta_recuperacion = False
        if estabilidad_actual > 60 and estado_actual in ['Vencido', 'Muy Vencido']:
            alerta_recuperacion = True
        
        tiempo_limite = int(mediana_gap_actual * 1.5)
        
        repeticiones = transiciones.get((perfil, perfil), 0)
        total_salidas = total_salidas_perfil.get(perfil, 0)
        prob_repeticion = (repeticiones / total_salidas * 100) if total_salidas > 0 else 0
        
        semana_activa = "Sí" if estado_actual in ['Vencido', 'Muy Vencido'] else "No"
        
        last_row = ultimo_suceso_perfil[perfil]
        
        estado_ultima_salida = "Normal"
        estabilidad_ultima_salida = 0.0
        
        if len(gaps) >= 1:
            gap_ultima_espera = gaps[-1]
            if len(gaps) > 1:
                medianas_previas = np.median(gaps[:-1])
            else:
                medianas_previas = gap_ultima_espera 
            
            estado_ultima_salida = calcular_estado_actual(gap_ultima_espera, medianas_previas)
            
            if len(gaps) > 1:
                estados_previos = [calcular_estado_actual(g, medianas_previas) for g in gaps[:-1]]
                mv_previos = estados_previos.count('Muy Vencido')
                estabilidad_ultima_salida = ((len(estados_previos) - mv_previos) / len(estados_previos) * 100)
            else:
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
            'Estado Ultima Salida': estado_ultima_salida,
            'Estabilidad Ultima Salida': round(estabilidad_ultima_salida, 1)
        })
        
    df_stats = pd.DataFrame(analisis_perfiles)
    return df_stats, transiciones, ultimo_perfil_global

# --- MOTORES DE PREDICCION ---

def obtener_prediccion_numeros_lista(df_stats, transiciones, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos):
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
        trans_count = transiciones.get((ultimo_perfil, p), 0)
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
    df_cands = df_cands.sort_values(['Score', 'Temp_Score'], ascending=[False, False]).drop_duplicates(subset=['Numero'])
    df_cands = df_cands.sort_values(by=['Score', 'Temp_Score', 'Gap_Num'], ascending=[False, False, False])
    return df_cands.head(30)['Numero'].tolist()

def generar_sugerencia_fusionada(df_stats, transiciones, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos):
    st.subheader("🤖 Sugerencia Inteligente Fusionada")
    st.markdown("El sistema analiza patrones de corrección ('Doble Fallo') y estabilidad para priorizar números.")
    
    st.markdown("### 🚨 Detalle de Alertas Activas")
    st.markdown("Información detallada del estado histórico de la combinación:")
    
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
            
            nums_alerta = [f"{d}{u}" for d in decenas_cumplen for u in unidades_cumplen]
            
            gap = row_alert['Gap Actual']
            med = row_alert['Mediana Gap']
            estado = row_alert['Estado Actual']
            estabilidad_val = row_alert['Estabilidad']
            
            ult_fecha = row_alert['Última Fecha']
            ult_num = row_alert['Último Numero']
            ult_sorteo = row_alert['Último Sorteo']
            ult_estado_real = row_alert['Estado Ultima Salida'] 
            ult_estabilidad_real = row_alert['Estabilidad Ultima Salida']
            
            time_str = ""
            if estado == "Normal":
                falta = int(med - gap)
                time_str = f"🟢 Faltan {falta} días"
            elif estado == "Vencido":
                falta_mv = int((med * 1.5) - gap)
                exceso = int(gap - med)
                time_str = f"🟠 Exceso {exceso} días (Faltan {falta_mv} para Muy Vencido)"
            elif estado == "Muy Vencido":
                exceso = int(gap - (med * 1.5))
                time_str = f"🔴 +{exceso} días exceso"
            
            ult_fecha_str = ult_fecha.strftime('%d/%m/%Y')
            
            st.markdown(f"**Perfil Alertado: `{perfil_name}`**")
            st.markdown(f"📍 **Estado Actual:** {estado} | ⏳ {time_str}")
            st.markdown(f"📊 **Estabilidad Actual:** {estabilidad_val}%")
            st.markdown("---")
            st.markdown(f"🔙 **Última vez que salió este perfil:**")
            st.markdown(f"- 📅 Fecha: **{ult_fecha_str}** ({ult_sorteo})")
            st.markdown(f"- 🔢 Número: **{ult_num:02d}**")
            st.markdown(f"- 🏷️ **Salió con Estado:** `{ult_estado_real}`") 
            st.markdown(f"- 📈 **Estabilidad en ese momento:** `{ult_estabilidad_real}%`")
            
            st.markdown(f"**Decenas '{ed_req}':** `{decenas_cumplen}` | **Unidades '{eu_req}':** `{unidades_cumplen}`")
            
            if nums_alerta:
                st.success(f"🔢 Números que forman esta alerta ({len(nums_alerta)}): {' - '.join(nums_alerta)}")
            else:
                st.warning("No se encontraron números que cumplan estrictamente.")
            st.markdown("---")
    else:
        st.info("No hay alertas de recuperación activas para esta fecha.")

    st.markdown("### 🎲 Top 30 Números Sugeridos")
    lista_nums = obtener_prediccion_numeros_lista(df_stats, transiciones, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos)
    
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
        trans_count = transiciones.get((ultimo_perfil, p), 0)
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
            <div style="font-size: 0.85em; font-weight:bold; color:{color_dec};">
                {ed_display}
            </div>
            <div style="font-size: 0.85em; font-weight:bold; color:{color_uni};">
                {eu_display}
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- BACKTEST ---
def ejecutar_backtest(df_full, dias_atras):
    hoy = datetime.now().date()
    resultados = []
    aciertos = 0
    total_sorteos = 0
    
    df_cache_full = obtener_historial_perfiles_cacheado(df_full, RUTA_CACHE)
    estabilidad_digitos = calcular_estabilidad_historica_digitos(df_full)
    
    st.info(f"Simulando {dias_atras} días...")
    progress_bar = st.progress(0)
    
    for i in range(dias_atras):
        current_date = hoy - timedelta(days=dias_atras - i - 1)
        fecha_ref = pd.to_datetime(current_date)
        
        for sorteo_tipo in ['T', 'N']:
            if sorteo_tipo == 'T':
                mask_datos = (df_full['Fecha'] < fecha_ref)
            else:
                mask_datos = (df_full['Fecha'] < fecha_ref) | ((df_full['Fecha'] == fecha_ref) & (df_full['Tipo_Sorteo'] == 'T'))
            
            df_disponible = df_full[mask_datos].copy()
            mask_real = (df_full['Fecha'] == fecha_ref) & (df_full['Tipo_Sorteo'] == sorteo_tipo) & (df_full['Posicion'] == 'Fijo')
            
            if mask_real.sum() == 0: continue
            resultado_real = df_full[mask_real]['Numero'].iloc[0]
            total_sorteos += 1
            
            if sorteo_tipo == 'T':
                mask_cache = (df_cache_full['Fecha'] < fecha_ref)
            else:
                mask_cache = (df_cache_full['Fecha'] < fecha_ref) | ((df_cache_full['Fecha'] == fecha_ref) & (df_cache_full['Sorteo'] == 'Tarde'))
            
            df_cache_sim = df_cache_full[mask_cache].copy()
            
            if df_disponible.empty or df_cache_sim.empty: continue
            
            df_oport_dec, df_oport_uni = analizar_oportunidad_por_digito(df_disponible, fecha_ref)
            df_stats, transiciones, ultimo_perfil = analizar_estadisticas_perfiles(df_cache_sim, fecha_ref)
            prediccion = obtener_prediccion_numeros_lista(df_stats, transiciones, ultimo_perfil, df_oport_dec, df_oport_uni, df_cache_sim, fecha_ref, estabilidad_digitos)
            
            if resultado_real in prediccion: aciertos += 1
            
            resultados.append({
                'Fecha': current_date, 'Sorteo': 'Tarde' if sorteo_tipo == 'T' else 'Noche',
                'Real': resultado_real, 'Acierto': '✅' if resultado_real in prediccion else '❌'
            })
        
        progress_bar.progress((i + 1) / dias_atras)
        
    progress_bar.empty()
    return pd.DataFrame(resultados), aciertos, total_sorteos

# --- MAIN ---
def main():
    st.sidebar.header("⚙️ Opciones")
    
    # --- FORMULARIO AGREGAR SORTEO ---
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
                st.success("¡Guardado!")
                time.sleep(1)
                st.cache_resource.clear()
                st.rerun()
            except Exception as err: st.error(f"Error: {err}")

    # --- CARGA DE DATOS ---
    df_full = cargar_datos_flotodo(RUTA_CSV)
    
    # --- VISUALIZACION LATERAL ULTIMOS RESULTADOS (CORREGIDO A STRICT LAST UPDATE) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Últimos Resultados")
    
    fecha_ref_default = pd.Timestamp.now(tz=None).normalize()
    target_sesion_default = "Tarde"
    
    if not df_full.empty:
        # Obtener la última fecha del CSV
        ultima_fecha_db = df_full['Fecha'].max()
        
        # Filtrar solo los datos de esa última fecha
        df_ultimos_dia = df_full[df_full['Fecha'] == ultima_fecha_db].copy()
        
        st.sidebar.markdown(f"**Fecha: {ultima_fecha_db.strftime('%d/%m/%Y')}**")
        
        # Buscar si existe Tarde y Noche para esa fecha
        existe_tarde = not df_ultimos_dia[df_ultimos_dia['Tipo_Sorteo'] == 'T'].empty
        existe_noche = not df_ultimos_dia[df_ultimos_dia['Tipo_Sorteo'] == 'N'].empty
        
        # Mostrar resultados
        if existe_tarde:
            row_t = df_ultimos_dia[df_ultimos_dia['Tipo_Sorteo'] == 'T'].iloc[0]
            num_fijo_t = df_ultimos_dia[(df_ultimos_dia['Tipo_Sorteo'] == 'T') & (df_ultimos_dia['Posicion'] == 'Fijo')]['Numero'].iloc[0]
            num_cent_t = df_ultimos_dia[(df_ultimos_dia['Tipo_Sorteo'] == 'T') & (df_ultimos_dia['Posicion'] == 'Centena')]['Numero'].iloc[0]
            num_c1_t = df_ultimos_dia[(df_ultimos_dia['Tipo_Sorteo'] == 'T') & (df_ultimos_dia['Posicion'] == '1er Corrido')]['Numero'].iloc[0]
            num_c2_t = df_ultimos_dia[(df_ultimos_dia['Tipo_Sorteo'] == 'T') & (df_ultimos_dia['Posicion'] == '2do Corrido')]['Numero'].iloc[0]
            
            st.sidebar.markdown(f"**☀️ Tarde**")
            st.sidebar.markdown(f"Fijo: `{int(num_fijo_t):02d}` | Cent: `{num_cent_t}`")
            st.sidebar.markdown(f"C1: `{num_c1_t}` | C2: `{num_c2_t}`")
            st.sidebar.markdown("---")
        
        if existe_noche:
            row_n = df_ultimos_dia[df_ultimos_dia['Tipo_Sorteo'] == 'N'].iloc[0]
            num_fijo_n = df_ultimos_dia[(df_ultimos_dia['Tipo_Sorteo'] == 'N') & (df_ultimos_dia['Posicion'] == 'Fijo')]['Numero'].iloc[0]
            num_cent_n = df_ultimos_dia[(df_ultimos_dia['Tipo_Sorteo'] == 'N') & (df_ultimos_dia['Posicion'] == 'Centena')]['Numero'].iloc[0]
            num_c1_n = df_ultimos_dia[(df_ultimos_dia['Tipo_Sorteo'] == 'N') & (df_ultimos_dia['Posicion'] == '1er Corrido')]['Numero'].iloc[0]
            num_c2_n = df_ultimos_dia[(df_ultimos_dia['Tipo_Sorteo'] == 'N') & (df_ultimos_dia['Posicion'] == '2do Corrido')]['Numero'].iloc[0]
            
            st.sidebar.markdown(f"**🌙 Noche**")
            st.sidebar.markdown(f"Fijo: `{int(num_fijo_n):02d}` | Cent: `{num_cent_n}`")
            st.sidebar.markdown(f"C1: `{num_c1_n}` | C2: `{num_c2_n}`")
            st.sidebar.markdown("---")

        # Lógica para definir la fecha de referencia por defecto (análisis automático)
        # Si en la última fecha falta la noche, analizar la noche.
        # Si están las dos, analizar la tarde del día siguiente (predicción futura lógica) o analizar el último dato real.
        # REQUERIMIENTO: Título según última actualización real.
        # Si existen las dos, el "último dato real" es Noche.
        
        # Determinar el último sorteo real registrado en el CSV cronológicamente
        df_fijos_sorted = df_full[df_full['Posicion'] == 'Fijo'].sort_values(by=['Fecha', 'Tipo_Sorteo'], ascending=[True, True])
        ultimo_registro = df_fijos_sorted.iloc[-1]
        
        fecha_ref_default = ultimo_registro['Fecha']
        target_sesion_default = "Tarde" if ultimo_registro['Tipo_Sorteo'] == 'T' else "Noche"

    else:
        st.sidebar.warning("No hay datos.")

    # --- CONFIGURACION DE FECHA DE ANALISIS ---
    modo_sorteo = st.sidebar.radio("Análisis:", ["General", "Tarde", "Noche"])
    modo_fecha = st.sidebar.radio("Fecha Ref:", ["Auto (Último Dato)", "Personalizado"])
    
    fecha_ref = fecha_ref_default
    target_sesion = target_sesion_default
    
    if modo_fecha == "Personalizado":
        fecha_ref = st.sidebar.date_input("Fecha:", datetime.now().date())
        fecha_ref = pd.to_datetime(fecha_ref)
        sesion_estado = st.sidebar.radio("Estado:", ["Antes de Tarde", "Después de Tarde"], horizontal=False)
        if sesion_estado == "Antes de Tarde": target_sesion = "Tarde"
        else: target_sesion = "Noche"

    if st.sidebar.button("🔄 Recargar"): st.rerun()
    
    if "Tarde" in modo_sorteo: df_analisis = df_full[df_full['Tipo_Sorteo'] == 'T'].copy()
    elif "Noche" in modo_sorteo: df_analisis = df_full[df_full['Tipo_Sorteo'] == 'N'].copy()
    else: df_analisis = df_full.copy()
    
    if df_analisis.empty: st.warning("Sin datos."); st.stop()

    # Filtros temporales
    if target_sesion == "Tarde": 
        df_backtest = df_analisis[df_analisis['Fecha'] < fecha_ref].copy()
    else: 
        df_backtest = df_analisis[(df_analisis['Fecha'] < fecha_ref) | ((df_analisis['Fecha'] == fecha_ref) & (df_analisis['Tipo_Sorteo'] == 'T'))].copy()

    # --- HISTORIAL DE COMBINACIONES ---
    st.header("📜 Historial de Combinaciones y Estados")
    st.markdown("Listado de sorteos ordenados de **más reciente a menos reciente** (Noche primero, luego Tarde).")
    
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
    df_estados_num, hist_num = get_full_state_dataframe(df_backtest, fecha_ref)
    df_oport_dec, df_oport_uni = analizar_oportunidad_por_digito(df_backtest, fecha_ref)
    
    st.header(f"🎯 Estado de Dígitos ({target_sesion} {fecha_ref.strftime('%d/%m')})")
    col1, col2 = st.columns(2)
    with col1: st.dataframe(df_oport_dec.sort_values('Punt. Base', ascending=False), hide_index=True)
    with col2: st.dataframe(df_oport_uni.sort_values('Punt. Base', ascending=False), hide_index=True)
        
    # 2. Análisis de Perfiles
    st.markdown("---")
    st.header("📅 Análisis de Perfiles (Motor Mejorado)")
    
    if st.button("🚀 Ejecutar Análisis", type="primary"):
        with st.spinner("Analizando..."):
            if not df_historial_perfiles_full.empty:
                # Ajustar caché según fecha referencia
                if target_sesion == "Tarde": 
                    df_historial_perfiles = df_historial_perfiles_full[df_historial_perfiles_full['Fecha'] < fecha_ref].copy()
                else: 
                    df_historial_perfiles = df_historial_perfiles_full[(df_historial_perfiles_full['Fecha'] < fecha_ref) | ((df_historial_perfiles_full['Fecha'] == fecha_ref) & (df_historial_perfiles_full['Sorteo'] == 'Tarde'))].copy()

                if not df_historial_perfiles.empty:
                    df_stats, transiciones, ultimo_perfil = analizar_estadisticas_perfiles(df_historial_perfiles, fecha_ref)
                    estabilidad_digitos = calcular_estabilidad_historica_digitos(df_backtest)
                    generar_sugerencia_fusionada(df_stats, transiciones, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos)
                    
                    # --- TABLA DE ESTADISTICAS RESTAURADA ---
                    st.markdown("---")
                    st.subheader("📊 Estadística de Perfiles (Completa)")
                    
                    cols_tabla = ['Perfil', 'Frecuencia', 'Última Fecha', 'Gap Actual', 'Mediana Gap', 
                                  'Estado Actual', 'Estabilidad', 'Tiempo Limite', 'Alerta', 
                                  'Prob Repeticiones %', 'Semana Activa']
                    
                    df_display = df_stats[cols_tabla].copy()
                    df_display['Última Fecha'] = df_display['Última Fecha'].dt.strftime('%d/%m/%Y')
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
