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
RUTA_BACKTEST = 'backtest_detalle.csv'
RUTA_HISTORICO = 'historico_predicciones.csv'

# --- CONFIGURACION DE LA PAGINA ---
st.set_page_config(
    page_title="Florida - Análisis de Sorteos",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌴 Florida - Análisis de Sorteos")
st.markdown("Motor con Backtest Real + Dashboard de Efectividad.")

# --- ESTADO DE SESIÓN ---
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'debug_logs' not in st.session_state:
    st.session_state.debug_logs = []
if 'invalid_dates_df' not in st.session_state:
    st.session_state.invalid_dates_df = None
if 'load_error' not in st.session_state:
    st.session_state.load_error = None
if 'rerun_counter' not in st.session_state:
    st.session_state.rerun_counter = 0
if 'last_rerun_time' not in st.session_state:
    st.session_state.last_rerun_time = 0
if 'perfil_distribuciones_cache' not in st.session_state:
    st.session_state.perfil_distribuciones_cache = {}

# --- CONTROL DE RERUN ---
current_time = time.time()
if current_time - st.session_state.last_rerun_time < 2:
    st.session_state.rerun_counter += 1
else:
    st.session_state.rerun_counter = 0
st.session_state.last_rerun_time = current_time

if st.session_state.rerun_counter > 3:
    st.error("🚨 Demasiadas recargas automáticas. Usa '🔄 Forzar Recarga' manualmente.")
    st.stop()

# --- FUNCIONES AUXILIARES ---

def remove_accents(input_str):
    if not isinstance(input_str, str): 
        return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def inicializar_archivo(ruta, columnas):
    if not os.path.exists(ruta):
        try:
            with open(ruta, 'w', encoding='utf-8') as f:
                f.write(";".join(columnas) + "\n")
            return True
        except Exception as e:
            st.error(f"❌ Error inicializando {ruta}: {e}")
            return False
    return True

def get_file_signature(filepath):
    try:
        if not os.path.exists(filepath):
            return None
        stat = os.stat(filepath)
        signature = f"{stat.st_size}_{int(stat.st_mtime)}"
        return signature
    except Exception as e:
        return None

def normalizar_numero(valor):
    try:
        if pd.isna(valor):
            return None
        return int(float(str(valor).strip()))
    except:
        return None

def numero_en_lista(numero, lista):
    try:
        num_normalizado = normalizar_numero(numero)
        if num_normalizado is None:
            return False
        for item in lista:
            if normalizar_numero(item) == num_normalizado:
                return True
        return False
    except:
        return False

def parse_fecha_safe(fecha_str):
    if pd.isna(fecha_str) or str(fecha_str).strip() == '':
        return None
    fecha_str = str(fecha_str).strip()
    formatos = ['%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d', '%d/%m/%y']
    for fmt in formatos:
        try:
            return pd.to_datetime(fecha_str, format=fmt, dayfirst=True)
        except:
            continue
    try:
        parsed = pd.to_datetime(fecha_str, dayfirst=True, errors='coerce')
        if pd.notna(parsed):
            return parsed
    except:
        pass
    return None

@st.cache_data(ttl=300, show_spinner=False)
def cargar_datos_flotodo(_ruta_csv, _file_signature):
    try:
        if not os.path.exists(_ruta_csv):
            inicializar_archivo(_ruta_csv, ["Fecha","Tipo_Sorteo","Centena","Fijo","Primer_Corrido","Segundo_Corrido"])
            return pd.DataFrame(columns=["Fecha","Tipo_Sorteo","Centena","Fijo","Primer_Corrido","Segundo_Corrido"]), None
        
        try:
            with open(_ruta_csv, 'r', encoding='latin-1') as f:
                primera_linea = f.readline()
            
            separador = ';' if ';' in primera_linea else (',' if ',' in primera_linea else '\t')
            
            df = pd.read_csv(_ruta_csv, sep=separador, encoding='latin-1', header=0, 
                           on_bad_lines='skip', dtype=str, skipinitialspace=True)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(), None
        except Exception as e:
            st.error(f"❌ Error al leer CSV: {e}")
            return pd.DataFrame(), None

        if df.empty: 
            return pd.DataFrame(), None
        
        df.columns = [str(c).strip() for c in df.columns]
        
        rename_map = {}
        for col in df.columns:
            c = str(col).strip().upper()
            if 'FECHA' in c: rename_map[col] = 'Fecha'
            elif any(x in c for x in ['NOCHE', 'TARDE', 'TIPO']): rename_map[col] = 'Tipo_Sorteo'
            elif 'CENTENA' in c: rename_map[col] = 'Centena'
            elif 'FIJO' in c and 'CORRIDO' not in c: rename_map[col] = 'Fijo'
            elif any(x in c for x in ['1ER', 'PRIMER', 'PRIMERO']): rename_map[col] = 'Primer_Corrido'
            elif any(x in c for x in ['2DO', 'SEGUNDO', 'SEGUND']): rename_map[col] = 'Segundo_Corrido'
        
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        
        cols_requeridas = ['Fecha', 'Tipo_Sorteo', 'Centena', 'Fijo', 'Primer_Corrido', 'Segundo_Corrido']
        cols_faltantes = [c for c in cols_requeridas if c not in df.columns]
        if cols_faltantes:
            for col in cols_faltantes:
                df[col] = ''
        
        invalid_dates_list = []
        fechas_procesadas = []
        
        for idx, row in df.iterrows():
            fecha_original = str(row['Fecha']).strip() if pd.notna(row['Fecha']) else ''
            fecha_parseada = parse_fecha_safe(fecha_original)
            
            if fecha_parseada is None and fecha_original != '':
                invalid_dates_list.append({
                    'Numero_Fila': idx, 'Fecha_Original': fecha_original,
                    'Centena': row.get('Centena', ''), 'Fijo': row.get('Fijo', ''),
                    'Primer_Corrido': row.get('Primer_Corrido', ''),
                    'Segundo_Corrido': row.get('Segundo_Corrido', ''),
                    'Problema': 'Fecha no reconocida'
                })
                fechas_procesadas.append(None)
            else:
                fechas_procesadas.append(fecha_parseada)
        
        df['Fecha'] = fechas_procesadas
        
        if invalid_dates_list:
            invalid_dates_df = pd.DataFrame(invalid_dates_list)
            df = df[df['Fecha'].notna()].copy()
        else:
            invalid_dates_df = None
        
        if df.empty:
            return pd.DataFrame(), invalid_dates_df
        
        df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper().map({
            'TARDE': 'T', 'T': 'T', 'TAR': 'T',
            'NOCHE': 'N', 'N': 'N', 'NOC': 'N', 'NOCTURNO': 'N'
        }).fillna('OTRO')
        df = df[df['Tipo_Sorteo'].isin(['T', 'N'])].copy()
        
        for col in ['Centena', 'Fijo', 'Primer_Corrido', 'Segundo_Corrido']:
            if col not in df.columns: df[col] = '0'
            df[col] = df[col].replace('', '0').fillna('0').astype(str).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        df_long = df.melt(id_vars=['Fecha', 'Tipo_Sorteo'], 
                         value_vars=['Centena', 'Fijo', 'Primer_Corrido', 'Segundo_Corrido'],
                         var_name='Posicion', value_name='Numero')
        
        pos_map = {'Centena': 'Centena', 'Fijo': 'Fijo', 'Primer_Corrido': '1er Corrido', 'Segundo_Corrido': '2do Corrido'}
        df_long['Posicion'] = df_long['Posicion'].map(pos_map)
        
        df_historial = df_long.dropna(subset=['Numero']).copy()
        df_historial['Numero'] = df_historial['Numero'].astype(int)
        
        draw_order_map = {'T': 0, 'N': 1}
        df_historial['draw_order'] = df_historial['Tipo_Sorteo'].map(draw_order_map)
        df_historial['sort_key'] = df_historial['Fecha'] + pd.to_timedelta(df_historial['draw_order'], unit='h')
        df_historial = df_historial.sort_values(by='sort_key').reset_index(drop=True)
        df_historial.drop(columns=['draw_order', 'sort_key'], inplace=True)
        
        if len(df_historial) > 1000:
            df_historial = df_historial.tail(1000).reset_index(drop=True)
        
        return df_historial, invalid_dates_df
        
    except Exception as e:
        st.error(f"❌ Error cargando datos: {e}")
        return pd.DataFrame(), None

def calcular_estado_actual(gap, limite_dinamico):
    if pd.isna(limite_dinamico) or limite_dinamico == 0 or limite_dinamico is None: 
        return "Normal"
    if gap > limite_dinamico: return "Muy Vencido"
    elif gap > (limite_dinamico * 0.66): return "Vencido"
    else: return "Normal"

def obtener_df_temperatura(contador):
    if not contador:
        return pd.DataFrame({'Dígito': range(10), 'Frecuencia': 0, 'Temperatura': '🟡 Tibio'})
    df = pd.DataFrame.from_dict(contador, orient='index', columns=['Frecuencia'])
    df = df.reset_index().rename(columns={'index': 'Dígito'})
    df = df.sort_values('Frecuencia', ascending=False).reset_index(drop=True)
    todos_digitos = pd.DataFrame({'Dígito': range(10)})
    df = todos_digitos.merge(df, on='Dígito', how='left').fillna({'Frecuencia': 0})
    df['Temperatura'] = '🟡 Tibio'
    if len(df) >= 3: df.loc[0:2, 'Temperatura'] = '🔥 Caliente'
    if len(df) >= 7: df.loc[6:9, 'Temperatura'] = '🧊 Frío'
    if len(df) >= 3: df.loc[3:5, 'Temperatura'] = '🟡 Tibio'
    return df

def analizar_oportunidad_por_digito(df_historial, fecha_referencia):
    if df_historial.empty: 
        return pd.DataFrame(), pd.DataFrame(), {}, {}, {}, {}
    
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
    
    df_hist_estado = df_base_fijos[df_base_fijos['Fecha'] <= fecha_referencia].copy()
    
    res_dec, res_uni = [], []
    gaps_dec = {}
    gaps_uni = {}
    freq_dec = {}
    freq_uni = {}
    
    for i in range(10):
        fechas_d = df_hist_estado[df_hist_estado['Numero'] // 10 == i]['Fecha'].sort_values()
        gap_d, prom_d = 0, 0
        if not fechas_d.empty:
            gaps = fechas_d.diff().dt.days.dropna()
            prom_d = gaps.median() if len(gaps) > 0 else 0
            gap_d = (fecha_referencia - fechas_d.max()).days
            gaps_dec[i] = gap_d
        else:
            gaps_dec[i] = 999
        ed = calcular_estado_actual(gap_d, prom_d)
        
        fechas_u = df_hist_estado[df_hist_estado['Numero'] % 10 == i]['Fecha'].sort_values()
        gap_u, prom_u = 0, 0
        if not fechas_u.empty:
            gaps = fechas_u.diff().dt.days.dropna()
            prom_u = gaps.median() if len(gaps) > 0 else 0
            gap_u = (fecha_referencia - fechas_u.max()).days
            gaps_uni[i] = gap_u
        else:
            gaps_uni[i] = 999
        eu = calcular_estado_actual(gap_u, prom_u)
        
        freq_dec[i] = contador_decenas.get(i, 0)
        freq_uni[i] = contador_unidades.get(i, 0)
        
        p_base_d = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}.get(ed, 0)
        p_base_u = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}.get(eu, 0)
        
        res_dec.append({'Dígito': i, 'Temperatura': mapa_temp_dec.get(i, '🟡 Tibio'), 'Estado': ed, 
                       'Punt. Base': p_base_d, 'Última Salida': fechas_d.max().strftime('%d/%m') if not fechas_d.empty else '-',
                       'Frecuencia': freq_dec[i]})
        res_uni.append({'Dígito': i, 'Temperatura': mapa_temp_uni.get(i, '🟡 Tibio'), 'Estado': eu, 
                       'Punt. Base': p_base_u, 'Última Salida': fechas_u.max().strftime('%d/%m') if not fechas_u.empty else '-',
                       'Frecuencia': freq_uni[i]})

    return pd.DataFrame(res_dec), pd.DataFrame(res_uni), gaps_dec, gaps_uni, freq_dec, freq_uni

def obtener_historial_perfiles_cacheado(df_full, ruta_cache=None):
    if df_full.empty: return pd.DataFrame()
    
    df_fijos = df_full[df_full['Posicion'] == 'Fijo'].copy()
    if df_fijos.empty: return pd.DataFrame()
    
    if len(df_fijos) > 1000:
        df_fijos = df_fijos.tail(1000).reset_index(drop=True)
    
    sort_val_map = {'T': 0, 'N': 1}
    df_fijos['sort_val'] = df_fijos['Tipo_Sorteo'].map(sort_val_map)
    df_fijos = df_fijos.sort_values(by=['Fecha', 'sort_val']).reset_index(drop=True)
    df_fijos.drop(columns=['sort_val'], inplace=True, errors='ignore')
    
    df_fijos['ID_Sorteo'] = df_fijos['Fecha'].dt.strftime('%Y-%m-%d') + "_" + df_fijos['Tipo_Sorteo']
    df_fijos = df_fijos.drop_duplicates(subset=['ID_Sorteo'], keep='last').reset_index(drop=True)
    
    df_cache = pd.DataFrame()
    use_file = ruta_cache and os.path.exists(ruta_cache)
    
    if use_file:
        try:
            df_cache = pd.read_csv(ruta_cache, parse_dates=['Fecha'], encoding='latin-1')
        except:
            df_cache = pd.DataFrame() 

    ids_en_cache = set()
    if not df_cache.empty and 'Fecha' in df_cache.columns and 'Sorteo' in df_cache.columns:
        sorteo_map_inv = {'Noche': 'N', 'Tarde': 'T', 'N': 'N', 'T': 'T'}
        df_cache['ID_Sorteo'] = df_cache['Fecha'].astype(str).str[:10] + "_" + df_cache['Sorteo'].astype(str).map(sorteo_map_inv)
        ids_en_cache = set(df_cache['ID_Sorteo'].dropna())
    
    df_nuevos = df_fijos[~df_fijos['ID_Sorteo'].isin(ids_en_cache)].copy()
    
    if df_nuevos.empty:
        cols_to_drop = [c for c in ['ID_Sorteo'] if c in df_cache.columns]
        if cols_to_drop:
            df_cache = df_cache.drop(columns=cols_to_drop)
        return df_cache
    
    df_nuevos = df_nuevos.sort_values(by=['Fecha', 'Tipo_Sorteo'])

    hist_decenas = defaultdict(list)
    hist_unidades = defaultdict(list)
    
    if not df_cache.empty and 'Fecha' in df_cache.columns:
        sort_val_inv = {'Tarde': 0, 'Noche': 1, 'T': 0, 'N': 1}
        df_cache['sort_val'] = df_cache['Sorteo'].astype(str).map(sort_val_inv).fillna(0)
        df_cache_sorted = df_cache.sort_values(by=['Fecha', 'sort_val'])
        for _, row in df_cache_sorted.iterrows():
            try:
                num = int(row['Numero'])
                fecha = pd.to_datetime(row['Fecha'])
                hist_decenas[num // 10].append(fecha)
                hist_unidades[num % 10].append(fecha)
            except:
                continue
            
    nuevos_registros = []
    
    for idx, row in df_nuevos.iterrows():
        try:
            fecha_actual = pd.to_datetime(row['Fecha'])
            num_actual = int(row['Numero'])
            tipo_actual = str(row['Tipo_Sorteo']).upper()
            
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
        except Exception as e:
            continue
    
    if nuevos_registros:
        df_nuevos_cache = pd.DataFrame(nuevos_registros)
        if not df_cache.empty:
            cols_to_drop = [c for c in ['ID_Sorteo', 'sort_val'] if c in df_cache.columns]
            df_final = pd.concat([df_cache.drop(columns=cols_to_drop, errors='ignore'), df_nuevos_cache], ignore_index=True)
        else:
            df_final = df_nuevos_cache
            
        if ruta_cache:
            try:
                df_final.to_csv(ruta_cache, index=False, encoding='latin-1')
            except:
                pass
        return df_final
    else:
        cols_to_drop = [c for c in ['ID_Sorteo'] if c in df_cache.columns]
        if cols_to_drop:
            df_cache = df_cache.drop(columns=cols_to_drop)
        return df_cache

def calcular_estabilidad_historica_digitos(df_full):
    if df_full.empty: return pd.DataFrame()
    df_fijos = df_full[df_full['Posicion'] == 'Fijo'].copy()
    if len(df_fijos) > 1000: df_fijos = df_fijos.tail(1000)
    resultados = []
    for i in range(10):
        fechas_d = df_fijos[df_fijos['Numero'] // 10 == i]['Fecha'].sort_values()
        if len(fechas_d) > 1:
            gaps = fechas_d.diff().dt.days.dropna()
            if len(gaps) > 0:
                med = gaps.median()
                excesos = sum(g > (med * 1.5) for g in gaps)
                estabilidad = 100 - (excesos / len(gaps) * 100)
            else: estabilidad = 50
        else: estabilidad = 50
        resultados.append({'Digito': i, 'Tipo': 'Decena', 'EstabilidadHist': round(estabilidad, 1)})
        
        fechas_u = df_fijos[df_fijos['Numero'] % 10 == i]['Fecha'].sort_values()
        if len(fechas_u) > 1:
            gaps = fechas_u.diff().dt.days.dropna()
            if len(gaps) > 0:
                med = gaps.median()
                excesos = sum(g > (med * 1.5) for g in gaps)
                estabilidad = 100 - (excesos / len(gaps) * 100)
            else: estabilidad = 50
        else: estabilidad = 50
        resultados.append({'Digito': i, 'Tipo': 'Unidad', 'EstabilidadHist': round(estabilidad, 1)})
    return pd.DataFrame(resultados)

def pre_calcular_distribuciones_perfiles(df_historial_perfiles):
    if df_historial_perfiles.empty: return {}
    
    distribuciones = {}
    perfiles_unicos = df_historial_perfiles['Perfil'].unique()
    
    for perfil in perfiles_unicos:
        df_perfil = df_historial_perfiles[df_historial_perfiles['Perfil'] == perfil].copy()
        
        if df_perfil.empty:
            distribuciones[perfil] = {'Normal': 33.3, 'Vencido': 33.3, 'Muy Vencido': 33.3, 'Estado_Comun': 'Normal', 'porcentaje': 33.3}
            continue
        
        estados_calculados = []
        fechas = df_perfil['Fecha'].sort_values().tolist()
        
        for i, fecha_actual in enumerate(fechas):
            if i == 0:
                estados_calculados.append('Normal')
                continue
            
            fechas_previas = fechas[:i]
            gaps = []
            for j in range(1, len(fechas_previas)):
                gap = (fechas_previas[j] - fechas_previas[j-1]).days
                if gap >= 0: gaps.append(gap)
            
            if len(gaps) == 0:
                estados_calculados.append('Normal')
                continue
            
            if len(gaps) >= 4:
                limite_dinamico = int(np.percentile(gaps, 75))
            else:
                limite_dinamico = int(np.median(gaps) * 2)
            
            gap_actual = (fecha_actual - fechas_previas[-1]).days
            estado = calcular_estado_actual(gap_actual, limite_dinamico)
            estados_calculados.append(estado)
        
        if len(estados_calculados) > 0:
            contador = Counter(estados_calculados)
            total = len(estados_calculados)
            distribucion = {estado: (count / total * 100) for estado, count in contador.items()}
            estado_comun = max(distribucion.keys(), key=lambda k: distribucion[k])
            if distribucion[estado_comun] < 60:
                estado_comun = 'Ninguno'
            
            distribuciones[perfil] = {
                'Normal': distribucion.get('Normal', 0),
                'Vencido': distribucion.get('Vencido', 0),
                'Muy Vencido': distribucion.get('Muy Vencido', 0),
                'Estado_Comun': estado_comun,
                'porcentaje': distribucion.get(estado_comun, 0)
            }
        else:
            distribuciones[perfil] = {'Normal': 33.3, 'Vencido': 33.3, 'Muy Vencido': 33.3, 'Estado_Comun': 'Normal', 'porcentaje': 33.3}
    
    return distribuciones

def calcular_p75_perfil(df_historial_perfiles, perfil_objetivo):
    if df_historial_perfiles.empty: return 0
    df_perfil = df_historial_perfiles[df_historial_perfiles['Perfil'] == perfil_objetivo].copy()
    if len(df_perfil) < 2: return 0
    fechas = df_perfil['Fecha'].sort_values()
    gaps = fechas.diff().dt.days.dropna()
    return int(np.percentile(gaps, 75)) if len(gaps) >= 4 else int(gaps.median() * 2) if len(gaps) > 0 else 0

def analizar_estadisticas_perfiles(df_historial_perfiles, fecha_referencia, distribuciones_cache=None):
    if df_historial_perfiles.empty:
        return pd.DataFrame(), Counter(), None
    
    if distribuciones_cache is None:
        distribuciones_cache = pre_calcular_distribuciones_perfiles(df_historial_perfiles)
        
    historial_fechas_perfiles = defaultdict(list)
    ultimo_suceso_perfil = {}
    transiciones = Counter()
    ultimo_perfil_global = None
    
    sort_val = {'Tarde': 0, 'Noche': 1, 'T': 0, 'N': 1}
    df_historial_perfiles = df_historial_perfiles.copy()
    df_historial_perfiles['sort_val'] = df_historial_perfiles['Sorteo'].astype(str).map(sort_val).fillna(0)
    df_historial_perfiles = df_historial_perfiles.sort_values(by=['Fecha', 'sort_val'])

    for _, row in df_historial_perfiles.iterrows():
        perfil = str(row['Perfil'])
        fecha = pd.to_datetime(row['Fecha'])
        numero = int(row['Numero'])
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
            diff = (fechas_ordenadas[k] - fechas_ordenadas[k-1]).days
            if diff >= 0: gaps.append(diff)
        
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
        estabilidad_ultima_salida = 100.0
        exceso_ultima_salida = 0
        
        if len(gaps) >= 1:
            gap_ultima_espera = gaps[-1]
            if len(gaps) > 1:
                gaps_prev = gaps[:-1]
                if len(gaps_prev) >= 4: 
                    lim_prev = int(np.percentile(gaps_prev, 75))
                else: 
                    lim_prev = int(np.median(gaps_prev) * 2) if gaps_prev else 0
                estado_ultima_salida = calcular_estado_actual(gap_ultima_espera, lim_prev)
                
                if estado_ultima_salida == "Muy Vencido" and lim_prev > 0:
                    exceso_ultima_salida = int(gap_ultima_espera - lim_prev)
                
                ests_prev = [calcular_estado_actual(g, lim_prev) for g in gaps_prev] if gaps_prev else []
                mv_prev = ests_prev.count('Muy Vencido')
                estabilidad_ultima_salida = ((len(ests_prev) - mv_prev) / len(ests_prev) * 100) if ests_prev else 100.0
            else:
                estado_ultima_salida = "Normal"
                estabilidad_ultima_salida = 100.0
        
        distribucion_estados = distribuciones_cache.get(perfil, {'Normal': 33.3, 'Vencido': 33.3, 'Muy Vencido': 33.3, 'Estado_Comun': 'Normal', 'porcentaje': 33.3})
        estado_ultima_salida_str = str(estado_ultima_salida)
        porc_ultima_salida = distribucion_estados.get(estado_ultima_salida_str, 33.3)
        
        fue_atipica = porc_ultima_salida < 20
        p75_perfil = calcular_p75_perfil(df_historial_perfiles, perfil)
        dias_desde_ultima_salida = (fecha_referencia - ultima_fecha).days if pd.notna(ultima_fecha) else 999
        dentro_enfriamiento = dias_desde_ultima_salida < p75_perfil if p75_perfil > 0 else False
        es_estado_comun = distribucion_estados.get(estado_actual, 0) > 60
        
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
            'Exceso Ultima Salida': exceso_ultima_salida,
            'Distribucion_Estados': distribucion_estados,
            'Fue_Atipica': fue_atipica,
            'P75_Perfil': p75_perfil,
            'Dias_Desde_Ultima': dias_desde_ultima_salida,
            'Dentro_Enfriamiento': dentro_enfriamiento,
            'Es_Estado_Comun': es_estado_comun,
            'Porc_Ultima_Salida': porc_ultima_salida
        })
    
    df_stats = pd.DataFrame(analisis_perfiles)
    return df_stats, transiciones, ultimo_perfil_global

def obtener_prediccion_numeros_lista(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos, gaps_dec, gaps_uni, distribuciones_perfiles, freq_dec, freq_uni, debug_mode=False, debug_logs=None):
    if df_stats.empty: 
        return list(range(10, 50))
    
    map_est_dec = {}
    map_est_uni = {}
    
    if not estabilidad_digitos.empty:
        dec_df = estabilidad_digitos[estabilidad_digitos['Tipo']=='Decena']
        uni_df = estabilidad_digitos[estabilidad_digitos['Tipo']=='Unidad']
        if not dec_df.empty: 
            map_est_dec = dec_df.set_index('Digito')['EstabilidadHist'].to_dict()
        if not uni_df.empty: 
            map_est_uni = uni_df.set_index('Digito')['EstabilidadHist'].to_dict()
    
    scores = []
    for _, row in df_stats.iterrows():
        p = str(row['Perfil'])
        score = 0
        estado = str(row['Estado Actual'])
        
        if row['Alerta'] == '⚠️ RECUPERAR': score += 150
        elif estado == 'Vencido': score += 70
        elif estado == 'Normal': score += 50
        elif estado == 'Muy Vencido': score += 100
        
        score += row['Estabilidad'] * 0.5
        if ultimo_perfil:
            trans_count = transizioni.get((str(ultimo_perfil), p), 0)
            score += trans_count * 10
        if row.get('Fue_Atipica', False): score -= 50
        if row.get('Dentro_Enfriamiento', False): score -= 30
        if row.get('Es_Estado_Comun', False): score += 30
        
        scores.append({'Perfil': p, 'Score': int(score), 'Estado': estado})
    
    df_scores = pd.DataFrame(scores).sort_values('Score', ascending=False)
    
    top_7 = df_scores.head(7)
    
    map_estado_dec = df_oport_dec.set_index('Dígito')['Estado'].to_dict() if not df_oport_dec.empty else {}
    map_estado_uni = df_oport_uni.set_index('Dígito')['Estado'].to_dict() if not df_oport_uni.empty else {}
    
    df_hist_nums = {}
    if not df_historial_perfiles.empty and 'Numero' in df_historial_perfiles.columns:
        df_hist_nums = df_historial_perfiles.groupby('Numero')['Fecha'].max().to_dict()
    
    candidatos_totales = []
    map_temp_dec = df_oport_dec.set_index('Dígito')['Temperatura'].to_dict() if not df_oport_dec.empty else {}
    map_temp_uni = df_oport_uni.set_index('Dígito')['Temperatura'].to_dict() if not df_oport_uni.empty else {}
    temp_val = {'🔥 Caliente': 3, '🟡 Tibio': 2, '🧊 Frío': 1}
    
    for _, row in top_7.iterrows():
        perfil = str(row['Perfil'])
        partes = perfil.split('-')
        if len(partes) != 2: 
            continue
        ed_req, eu_req = partes[0], partes[1]
        
        decenas_estado = [d for d in range(10) if map_estado_dec.get(d) == ed_req]
        unidades_estado = [u for u in range(10) if map_estado_uni.get(u) == eu_req]
        
        if not decenas_estado:
            decenas_estado = list(range(10))
        if not unidades_estado:
            unidades_estado = list(range(10))
        
        total_combinaciones = len(decenas_estado) * len(unidades_estado)
        porc_perfil = distribuciones_perfiles.get(perfil, {}).get('porcentaje', 0)
        
        for d in decenas_estado:
            for u in unidades_estado:
                num = int(f"{d}{u}")
                last_seen = df_hist_nums.get(num, pd.Timestamp('2000-01-01'))
                gap_n = (fecha_ref - last_seen).days if isinstance(last_seen, pd.Timestamp) else 999
                
                score_base = int(row['Score'])
                
                temp_d = temp_val.get(map_temp_dec.get(d, '🟡 Tibio'), 2)
                temp_u = temp_val.get(map_temp_uni.get(u, '🟡 Tibio'), 2)
                
                est_d = map_est_dec.get(d, 50)
                est_u = map_est_uni.get(u, 50)
                
                gap_bonus = min(gap_n / 10, 20)
                
                bonus_salidor = 0
                gap_d = gaps_dec.get(d, 0)
                gap_u = gaps_uni.get(u, 0)
                
                if map_estado_dec.get(d) in ['Vencido', 'Muy Vencido'] and freq_dec.get(d, 0) >= 8:
                    bonus_salidor += min(gap_d / 5, 20)
                    if map_estado_dec.get(d) == 'Muy Vencido':
                        bonus_salidor += 10
                
                if map_estado_uni.get(u) in ['Vencido', 'Muy Vencido'] and freq_uni.get(u, 0) >= 8:
                    bonus_salidor += min(gap_u / 5, 20)
                    if map_estado_uni.get(u) == 'Muy Vencido':
                        bonus_salidor += 10
                
                if (map_estado_dec.get(d) in ['Vencido', 'Muy Vencido'] and freq_dec.get(d, 0) >= 8) and \
                   (map_estado_uni.get(u) in ['Vencido', 'Muy Vencido'] and freq_uni.get(u, 0) >= 8):
                    bonus_salidor += 25
                
                bonus_escasez = 0
                if porc_perfil >= 40:
                    bonus_escasez += 15
                    if total_combinaciones < 20:
                        bonus_escasez += 20
                    if total_combinaciones < 10:
                        bonus_escasez += 10
                
                if porc_perfil >= 40:
                    if freq_uni.get(u, 0) >= 8 and gap_u > 15:
                        bonus_escasez += 25
                    if freq_dec.get(d, 0) >= 8 and gap_d > 15:
                        bonus_escasez += 25
                    
                    if porc_perfil >= 40 and total_combinaciones < 20 and \
                       (freq_uni.get(u, 0) >= 8 and gap_u > 15):
                        bonus_escasez += 30
                
                bonus_estabilidad = 0
                if map_estado_dec.get(d) == 'Normal' and freq_dec.get(d, 0) < 8:
                    bonus_estabilidad += 5
                if map_estado_uni.get(u) == 'Normal' and freq_uni.get(u, 0) < 8:
                    bonus_estabilidad += 5
                
                bonus_historial = 0
                if num in df_hist_nums:
                    freq_num = sum(1 for n in df_hist_nums.keys() if n == num)
                    if freq_num >= 3:
                        bonus_historial += 10
                    if gap_n > 20:
                        bonus_historial += min(gap_n / 10, 15)
                
                temp_score = temp_d + temp_u + (est_d + est_u) / 20 + gap_bonus + bonus_salidor + bonus_escasez + bonus_estabilidad + bonus_historial
                
                candidatos_totales.append({
                    'Numero': num, 
                    'Perfil': perfil, 
                    'Score': score_base,
                    'Gap_Num': gap_n, 
                    'Temp_Score': temp_score,
                    'Bonus_Salidor': bonus_salidor,
                    'Bonus_Escasez': bonus_escasez
                })
    
    if not candidatos_totales:
        return list(range(0, 40))
    
    df_cands = pd.DataFrame(candidatos_totales)
    
    max_por_perfil = 20
    df_cands = df_cands.sort_values(['Score', 'Temp_Score', 'Gap_Num', 'Numero'], ascending=[False, False, False, True])
    
    candidatos_finales = []
    perfil_counter = Counter()
    for _, row in df_cands.iterrows():
        perfil = row['Perfil']
        if perfil_counter[perfil] < max_por_perfil:
            candidatos_finales.append(row['Numero'])
            perfil_counter[perfil] += 1
        if len(candidatos_finales) >= 40:
            break
    
    if len(candidatos_finales) < 40:
        for num in range(0, 100):
            if num not in candidatos_finales:
                candidatos_finales.append(num)
            if len(candidatos_finales) >= 40:
                break
    
    resultado = candidatos_finales[:40]
    
    if debug_mode and debug_logs is not None:
        debug_logs.append(f"📋 Top 40: {resultado}")
    
    return resultado

def generar_sugerencia_fusionada(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos, gaps_dec, gaps_uni, distribuciones_perfiles, freq_dec, freq_uni):
    st.subheader("🤖 Sugerencia Inteligente Fusionada")
    
    with st.expander("📖 ¿Cómo funciona la lógica?", expanded=False):
        st.markdown("""
        **Estados (P75):** Normal | Vencido | Muy Vencido
        **TOP 7:** Usa 7 de 9 perfiles (78% cobertura)
        **TOP 40:** Genera 40 números en lugar de 30
        **Bonus Salidor en Deuda:** Dígitos frecuentes con gap alto
        **Bonus Escasez:** Perfiles frecuentes con pocas combinaciones
        """)

    st.markdown("### 🚨 Detalle de Alertas Activas")
    map_estado_dec = df_oport_dec.set_index('Dígito')['Estado'].to_dict() if not df_oport_dec.empty else {}
    map_estado_uni = df_oport_uni.set_index('Dígito')['Estado'].to_dict() if not df_oport_uni.empty else {}
    alertas_activas = df_stats[df_stats['Alerta'] == '⚠️ RECUPERAR'].copy() if not df_stats.empty else pd.DataFrame()
    
    if not alertas_activas.empty:
        for _, row_alert in alertas_activas.iterrows():
            perfil_name = str(row_alert['Perfil'])
            partes = perfil_name.split('-')
            if len(partes) != 2: continue
            ed_req, eu_req = partes[0], partes[1]
            decenas_cumplen = [d for d in range(10) if map_estado_dec.get(d) == ed_req]
            unidades_cumplen = [u for u in range(10) if map_estado_uni.get(u) == eu_req]
            nums_alerta = [f"{int(f'{d}{u}'):02d}" for d in decenas_cumplen for u in unidades_cumplen]
            
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
                falta = int(med - gap) if med > gap else 0
                time_str = f"🟢 Faltan ~{falta} días para Vencido"
            elif estado == "Vencido":
                falta_mv = int(tiempo_limite - gap) if tiempo_limite > gap else 0
                exceso = int(gap - med)
                time_str = f"🟠 Exceso: {exceso} días | Faltan {falta_mv} para Muy Vencido"
            elif estado == "Muy Vencido":
                exceso = int(gap - tiempo_limite) if tiempo_limite > 0 else gap
                time_str = f"🔴 +{exceso} días sobre límite P75"
            
            fue_atipica = row_alert.get('Fue_Atipica', False)
            dentro_enfriamiento = row_alert.get('Dentro_Enfriamiento', False)
            es_comun = row_alert.get('Es_Estado_Comun', False)
            porc_ultima = row_alert.get('Porc_Ultima_Salida', 0)
            distribucion = row_alert.get('Distribucion_Estados', {})
            
            with st.container(border=True):
                st.markdown(f"**Perfil Alertado: `{perfil_name}`**")
                st.markdown(f"📍 **Estado Actual:** `{estado}` | ⏳ {time_str}")
                st.markdown(f"📉 **Estabilidad Actual:** `{row_alert['Estabilidad']}%`")
                st.markdown(f"📉 **Estabilidad Última Salida:** `{ult_estabilidad}%`")
                st.markdown(f"📊 **Datos:** Gap: **{gap} días** | Límite P75: **{tiempo_limite} días** | Mediana: **{med} días**")
                
                if distribucion:
                    dist_str = " | ".join([f"{k}: {v:.1f}%" for k, v in distribucion.items() if isinstance(v, (int, float))])
                    st.markdown(f"📈 **Distribución Histórica:** {dist_str}")
                
                if fue_atipica:
                    st.warning(f"⚠️ **Estado Atípico Detectado:** La última salida fue en estado '{ult_estado}' (solo {porc_ultima:.1f}% del histórico)")
                if dentro_enfriamiento:
                    st.warning(f"⏳ **Dentro de Enfriamiento:** Faltan {row_alert['P75_Perfil'] - row_alert['Dias_Desde_Ultima']} días")
                if es_comun:
                    st.success(f"✅ **Estado Común:** Este perfil suele salir en estado '{estado}'")
                
                st.markdown(f"🔢 **Decenas ({ed_req}):** `{decenas_cumplen}` | **Unidades ({eu_req}):** `{unidades_cumplen}`")
                
                if nums_alerta:
                    with st.expander(f"🔢 Ver {len(nums_alerta)} números de esta alerta"):
                        chunks = [nums_alerta[x:x+10] for x in range(0, len(nums_alerta), 10)]
                        for chunk in chunks: st.write(f"`{' '.join(chunk)}`")
                
                st.info(f"🔙 **Historia de la última salida de este perfil:**")
                try:
                    fecha_fmt = ult_fecha.strftime('%d/%m/%Y') if pd.notna(ult_fecha) else 'N/A'
                    st.markdown(f"- Fecha: **{fecha_fmt}**")
                except:
                    st.markdown(f"- Fecha: **{ult_fecha}**")
                st.markdown(f"- Estado en ese momento: **{ult_estado}**")
                st.markdown(f"- Estabilidad previa: **{ult_estabilidad}%**")
                if ult_estado == "Muy Vencido" and ult_exceso > 0:
                    st.markdown(f"- Días en exceso: **{ult_exceso} días**")
            
            st.markdown("---")
    else:
        st.info("✅ No hay alertas activas.")
    
    st.markdown("### 🎲 Top 40 Números Sugeridos")
    lista_nums = obtener_prediccion_numeros_lista(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos, gaps_dec, gaps_uni, distribuciones_perfiles, freq_dec, freq_uni)
    
    if not lista_nums:
        st.warning("⚠️ No se generaron candidatos.")
        return
    
    def get_state_color_hex(state):
        return {'Normal': '#22c55e', 'Vencido': '#f59e0b', 'Muy Vencido': '#ef4444'}.get(str(state), '#94a3b8')
    
    def shorten_state(text):
        return {"Muy Vencido": "M.Vencido", "Vencido": "Vencido", "Normal": "Normal"}.get(str(text), str(text))

    cols = st.columns(8)
    for idx, num in enumerate(lista_nums):
        try:
            d_int, u_int = int(num // 10), int(num % 10)
            ed, eu = map_estado_dec.get(d_int, "?"), map_estado_uni.get(u_int, "?")
            cols[idx % 8].markdown(f"""
            <div style="background-color:#1e1e1e; padding:12px; border-radius:10px; text-align:center; border:1px solid #444;">
                <h3 style="margin:0; color:#4ade80; font-size:1.4em;">{num:02d}</h3>
                <div style="font-size:0.8em; color:{get_state_color_hex(str(ed))};">{shorten_state(str(ed))}</div>
                <div style="font-size:0.8em; color:{get_state_color_hex(str(eu))};">{shorten_state(str(eu))}</div>
            </div>
            """, unsafe_allow_html=True)
        except: continue

def guardar_prediccion_en_historico(fecha, sorteo, top30, perfil_top, score_promedio, numero_real=None):
    archivo = RUTA_HISTORICO
    
    if numero_real is not None:
        estuvo_en_top30 = numero_real in top30
        acierto = '✅' if estuvo_en_top30 else '❌'
        posicion = str(top30.index(numero_real) + 1) if estuvo_en_top30 else 'N/A'
    else:
        acierto = 'PENDIENTE'
        posicion = 'PENDIENTE'
    
    sorteo_nombre = 'Tarde' if str(sorteo).upper() == 'T' else ('Noche' if str(sorteo).upper() == 'N' else str(sorteo))
    
    if hasattr(fecha, 'strftime'):
        fecha_str = fecha.strftime('%Y-%m-%d')
    else:
        fecha_str = str(fecha)[:10]
    
    existe = os.path.exists(archivo)
    
    try:
        with open(archivo, 'a', encoding='utf-8', newline='') as f:
            if not existe:
                f.write('Fecha,Sorteo,Top30,Perfil_Top,Score_Promedio,Numero_Real,Acierto,Posicion\n')
            
            linea = f'{fecha_str},{sorteo_nombre},"{top30}",{perfil_top},{score_promedio},{numero_real if numero_real else "PENDIENTE"},{acierto},{posicion}\n'
            f.write(linea)
            f.flush()
        
        return True
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return False

def leer_historico_predicciones():
    if not os.path.exists(RUTA_HISTORICO):
        return pd.DataFrame()
    
    try:
        with open(RUTA_HISTORICO, 'r', encoding='utf-8') as f:
            contenido = f.read()
        
        st.write("**📄 Contenido del archivo:**")
        st.code(contenido[:800])
        
        df = pd.read_csv(RUTA_HISTORICO, encoding='utf-8')
        
        if not df.empty:
            st.write(f"**📋 Columnas disponibles:** {list(df.columns)}")
        
        return df
    except Exception as e:
        st.error(f"❌ Error al leer: {e}")
        return pd.DataFrame()

def actualizar_resultados_pendientes(numero_real, fecha, sorteo):
    if not os.path.exists(RUTA_HISTORICO):
        return 0
    
    try:
        df = pd.read_csv(RUTA_HISTORICO, encoding='utf-8')
        if df.empty:
            return 0
        
        sorteo_nombre = 'Tarde' if str(sorteo).upper() == 'T' else ('Noche' if str(sorteo).upper() == 'N' else str(sorteo))
        
        if hasattr(fecha, 'strftime'):
            fecha_str = fecha.strftime('%Y-%m-%d')
        else:
            fecha_str = str(fecha)[:10]
        
        mask_pendiente = (
            (df['Fecha'].astype(str).str[:10] == fecha_str) & 
            (df['Sorteo'].astype(str) == sorteo_nombre) & 
            (df['Acierto'].astype(str) == 'PENDIENTE')
        )
        
        if mask_pendiente.any():
            actualizados = 0
            for idx in df[mask_pendiente].index:
                try:
                    top30_str = str(df.loc[idx, 'Top30']).strip('[]').strip('"')
                    top30 = [int(x.strip()) for x in top30_str.split(',') if x.strip()]
                    
                    estuvo_en_top30 = numero_real in top30
                    df.loc[idx, 'Numero_Real'] = str(numero_real)
                    df.loc[idx, 'Acierto'] = '✅' if estuvo_en_top30 else '❌'
                    df.loc[idx, 'Posicion'] = str(top30.index(numero_real) + 1) if estuvo_en_top30 else 'N/A'
                    actualizados += 1
                except Exception as e:
                    pass
            
            with open(RUTA_HISTORICO, 'w', encoding='utf-8', newline='') as f:
                f.write('Fecha,Sorteo,Top30,Perfil_Top,Score_Promedio,Numero_Real,Acierto,Posicion\n')
                for _, row in df.iterrows():
                    linea = f'{row["Fecha"]},{row["Sorteo"]},"{row["Top30"]}",{row["Perfil_Top"]},{row["Score_Promedio"]},{row["Numero_Real"]},{row["Acierto"]},{row["Posicion"]}\n'
                    f.write(linea)
            
            return actualizados
    except Exception as e:
        st.error(f"❌ Error: {e}")
    
    return 0

def mostrar_dashboard_efectividad():
    st.header("📊 Dashboard de Efectividad")
    st.markdown("Análisis automático de predicciones anteriores.")
    
    df = leer_historico_predicciones()
    
    with st.expander("🐛 DEBUG: Ver contenido del histórico", expanded=True):
        st.write(f"**📁 Archivo:** {RUTA_HISTORICO}")
        st.write(f"**✅ Archivo existe:** {os.path.exists(RUTA_HISTORICO)}")
        if os.path.exists(RUTA_HISTORICO):
            st.write(f"**📏 Tamaño:** {os.path.getsize(RUTA_HISTORICO)} bytes")
        st.write(f"**📊 Total filas:** {len(df)}")
        if len(df) > 0:
            st.write(f"**📈 Con resultado:** {len(df[df['Acierto'] != 'PENDIENTE'])}")
            st.write(f"**⏳ Pendientes:** {len(df[df['Acierto'] == 'PENDIENTE'])}")
            st.write(f"**📋 Columnas:** {list(df.columns)}")
            st.write("**📄 Contenido completo:**")
            st.dataframe(df, hide_index=True)
        else:
            st.info("ℹ️ El archivo está vacío")
    
    if df.empty:
        st.info("ℹ️ No hay predicciones guardadas aún. Ejecuta análisis primero.")
        return
    
    df_completas = df[df['Acierto'] != 'PENDIENTE'].copy()
    
    if len(df_completas) == 0:
        st.warning("⚠️ Aún no hay predicciones con resultado. Agrega el sorteo real después de generar una predicción.")
        return
    
    col1, col2, col3 = st.columns(3)
    total = len(df_completas)
    aciertos = len(df_completas[df_completas['Acierto'] == '✅'])
    efectividad = (aciertos / total * 100) if total > 0 else 0
    
    col1.metric("📊 Total Predicciones", total)
    col2.metric("✅ Aciertos", aciertos)
    col3.metric("📈 Efectividad General", f"{efectividad:.1f}%")
    
    st.markdown("---")
    
    st.subheader("📈 Efectividad por Perfil")
    if 'Perfil_Top' in df_completas.columns and df_completas['Perfil_Top'].notna().any():
        perfil_stats = df_completas.groupby('Perfil_Top').apply(
            lambda x: pd.Series({
                'Total': len(x),
                'Aciertos': len(x[x['Acierto'] == '✅']),
                'Efectividad': round(len(x[x['Acierto'] == '✅']) / len(x) * 100, 1) if len(x) > 0 else 0
            })
        ).reset_index()
        
        if not perfil_stats.empty and len(perfil_stats) > 0:
            perfil_stats = perfil_stats.sort_values('Efectividad', ascending=False)
            
            for _, row in perfil_stats.iterrows():
                color = "🟢" if row['Efectividad'] >= 40 else ("🟡" if row['Efectividad'] >= 25 else "🔴")
                st.markdown(f"{color} **{row['Perfil_Top']}**: {row['Efectividad']}% ({row['Aciertos']} de {row['Total']})")
    
    st.markdown("---")
    
    st.subheader("📈 Efectividad por Rango de Score")
    if 'Score_Promedio' in df_completas.columns and df_completas['Score_Promedio'].notna().any():
        score_dict = {}
        
        for label in ['<150', '150-200', '200-250', '>250']:
            if label == '<150':
                mask = df_completas['Score_Promedio'] < 150
            elif label == '150-200':
                mask = (df_completas['Score_Promedio'] >= 150) & (df_completas['Score_Promedio'] < 200)
            elif label == '200-250':
                mask = (df_completas['Score_Promedio'] >= 200) & (df_completas['Score_Promedio'] < 250)
            else:
                mask = df_completas['Score_Promedio'] >= 250
            
            total_en_rango = len(df_completas[mask])
            aciertos_en_rango = len(df_completas[mask & (df_completas['Acierto'] == '✅')])
            efectividad_en_rango = round(aciertos_en_rango / total_en_rango * 100, 1) if total_en_rango > 0 else 0
            
            score_dict[label] = {
                'Total': total_en_rango,
                'Aciertos': aciertos_en_rango,
                'Efectividad': efectividad_en_rango
            }
        
        for label, datos in score_dict.items():
            if datos['Total'] > 0:
                color = "🟢" if datos['Efectividad'] >= 40 else ("🟡" if datos['Efectividad'] >= 25 else "🔴")
                st.markdown(f"{color} **Score {label}**: {datos['Efectividad']}% ({datos['Aciertos']} de {datos['Total']})")
    
    with st.expander("📋 Ver Histórico Completo"):
        st.dataframe(df.sort_values('Fecha', ascending=False).head(50), hide_index=True, use_container_width=True)

def ejecutar_backtest(df_full, sorteos_objetivo, nombre_sesion, debug_mode=False):
    if df_full.empty: 
        st.warning(f"⚠️ No hay datos para backtest de {nombre_sesion}")
        return pd.DataFrame(), 0, 0, []
    
    df_fijos = df_full[df_full['Posicion'] == 'Fijo'].copy()
    if df_fijos.empty: 
        return pd.DataFrame(), 0, 0, []
    
    if len(df_fijos) > 1000: 
        df_fijos = df_fijos.tail(1000).reset_index(drop=True)
    
    df_fijos['sort_val'] = df_fijos['Tipo_Sorteo'].map({'T': 0, 'N': 1})
    df_fijos = df_fijos.sort_values(by=['Fecha', 'sort_val'], ascending=[False, False]).reset_index(drop=True)
    
    sorteos_a_procesar = min(sorteos_objetivo, len(df_fijos))
    if sorteos_a_procesar < 1: 
        return pd.DataFrame(), 0, 0, []
    
    resultados, aciertos, total_sorteos = [], 0, 0
    debug_logs = [] if debug_mode else []
    
    df_cache_full = obtener_historial_perfiles_cacheado(df_full)
    distribuciones_cache = pre_calcular_distribuciones_perfiles(df_cache_full) if not df_cache_full.empty else {}
    estabilidad_digitos_global = calcular_estabilidad_historica_digitos(df_full)
    
    progress_bar = st.progress(0)
    start_time = time.time()
    
    for i in range(sorteos_a_procesar):
        sorteo_actual = df_fijos.iloc[i]
        fecha_ref, tipo_sorteo = sorteo_actual['Fecha'], sorteo_actual['Tipo_Sorteo']
        resultado_real = int(sorteo_actual['Numero'])
        total_sorteos += 1
        
        df_historial = df_full[df_full['Fecha'] < fecha_ref].copy()
        if len(df_historial) > 1000: 
            df_historial = df_historial.tail(1000)
        
        if df_historial.empty:
            resultados.append({'Fecha': fecha_ref.strftime('%d/%m/%Y'), 'Sorteo': tipo_sorteo, 'Real': f"{resultado_real:02d}", 'En Top30': '⚠️'})
            progress_bar.progress((i + 1) / sorteos_a_procesar)
            continue
        
        df_oport_dec, df_oport_uni, gaps_dec, gaps_uni, freq_dec, freq_uni = analizar_oportunidad_por_digito(df_historial, fecha_ref)
        
        df_historial_perfiles = df_cache_full[df_cache_full['Fecha'] < fecha_ref].copy() if not df_cache_full.empty else pd.DataFrame()
        if df_historial_perfiles.empty:
            df_stats, transizioni, ultimo_perfil = pd.DataFrame(), Counter(), None
        else:
            df_stats, transizioni, ultimo_perfil = analizar_estadisticas_perfiles(df_historial_perfiles, fecha_ref, distribuciones_cache=distribuciones_cache)
        
        prediccion = obtener_prediccion_numeros_lista(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos_global, gaps_dec, gaps_uni, distribuciones_cache, freq_dec, freq_uni)
        
        es_acierto = numero_en_lista(resultado_real, prediccion)
        if es_acierto: 
            aciertos += 1
        
        if debug_mode:
            debug_logs.append({'Fecha': fecha_ref.strftime('%d/%m/%Y'), 'Sorteo': tipo_sorteo, 'Real': resultado_real, 'En Top30': '✅' if es_acierto else '❌'})
        
        resultados.append({'Fecha': fecha_ref.strftime('%d/%m/%Y'), 'Sorteo': tipo_sorteo, 'Real': f"{resultado_real:02d}", 'En Top30': '✅' if es_acierto else '❌'})
        
        elapsed = time.time() - start_time
        progress_bar.progress((i + 1) / sorteos_a_procesar)
        
        if elapsed > 120:
            st.warning("⚠️ Backtest excedió 120 segundos.")
            break
    
    progress_bar.empty()
    return pd.DataFrame(resultados), aciertos, total_sorteos, debug_logs

def mostrar_tabla_personalidad_perfiles(df_historial_perfiles):
    st.header("📊 Comportamiento Histórico de Perfiles")
    if df_historial_perfiles.empty:
        st.info("ℹ️ No hay datos suficientes.")
        return
    
    distribuciones = pre_calcular_distribuciones_perfiles(df_historial_perfiles)
    if not distribuciones:
        st.info("ℹ️ No hay perfiles.")
        return
    
    filas_tabla = []
    for perfil, dist in distribuciones.items():
        estado_comun = dist.get('Estado_Comun', 'Ninguno')
        recomendacion = "✅ Jugar en " + estado_comun if estado_comun in ['Normal', 'Vencido', 'Muy Vencido'] else "⚠️ Sin patrón"
        filas_tabla.append({
            'Perfil': perfil,
            'Normal': f"{dist.get('Normal', 0):.1f}%",
            'Vencido': f"{dist.get('Vencido', 0):.1f}%",
            'Muy Vencido': f"{dist.get('Muy Vencido', 0):.1f}%",
            'Estado Común': estado_comun,
            'Recomendación': recomendacion
        })
    
    df_tabla = pd.DataFrame(filas_tabla)
    st.dataframe(df_tabla, hide_index=True, use_container_width=True)
    st.caption(f"📊 {len(filas_tabla)} perfiles | últimos 1000 sorteos")

def main():
    st.sidebar.header("⚙️ Panel de Control")
    
    st.session_state.debug_mode = st.sidebar.checkbox("🐛 Modo Debug", value=st.session_state.debug_mode)
    
    if st.sidebar.button("🔄 Forzar Recarga", type="primary"):
        st.cache_data.clear()
        for key in list(st.session_state.keys()): del st.session_state[key]
        if os.path.exists(RUTA_CACHE):
            try: os.remove(RUTA_CACHE)
            except: pass
        time.sleep(0.5)
        st.rerun()
    
    file_signature = None
    if os.path.exists(RUTA_CSV):
        try:
            file_signature = get_file_signature(RUTA_CSV)
            file_size = os.path.getsize(RUTA_CSV)
            file_mod = datetime.fromtimestamp(os.path.getmtime(RUTA_CSV)).strftime('%d/%m %H:%M')
            st.sidebar.caption(f"📄 {RUTA_CSV}\n📏 {file_size} bytes\n🕐 {file_mod}")
        except Exception as e:
            st.sidebar.error(f"Error leyendo archivo: {e}")
            file_signature = None
    else:
        st.sidebar.warning(f"⚠️ {RUTA_CSV} no encontrado")
        inicializar_archivo(RUTA_CSV, ["Fecha","Tipo_Sorteo","Centena","Fijo","Primer_Corrido","Segundo_Corrido"])
        st.info(f"✅ Archivo creado. Agrega tu primer sorteo.")
        st.stop()
    
    df_full, invalid_dates_df = cargar_datos_flotodo(RUTA_CSV, file_signature)
    st.session_state.invalid_dates_df = invalid_dates_df
    
    if st.session_state.invalid_dates_df is not None and len(st.session_state.invalid_dates_df) > 0:
        st.warning(f"⚠️ {len(st.session_state.invalid_dates_df)} filas con fechas inválidas")
        st.dataframe(st.session_state.invalid_dates_df.head(10), hide_index=True)
    
    if st.session_state.load_error:
        st.error(f"❌ Error: {st.session_state.load_error}")
        st.markdown("---")
    
    with st.sidebar.expander("📝 Agregar Sorteo", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            f_nueva = st.date_input("Fecha", datetime.now().date(), key="date_input")
        with col_f2:
            ses = st.radio("Sesión", ["Tarde", "Noche"], horizontal=True, key="sesion_radio")
        
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            cent = st.number_input("Centena", 0, 999, 0, key="inp_cent")
            fij = st.number_input("Fijo", 0, 99, 0, key="inp_fijo")
        with col_n2:
            c1 = st.number_input("C1", 0, 99, 0, key="inp_c1")
            c2 = st.number_input("C2", 0, 99, 0, key="inp_c2")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("💾 Guardar", type="primary", use_container_width=True):
                s_code = {"Tarde": "T", "Noche": "N"}[ses]
                line = f"{f_nueva.strftime('%d/%m/%Y')};{s_code};{int(cent)};{int(fij)};{int(c1)};{int(c2)}\n"
                try:
                    with open(RUTA_CSV, 'a', encoding='latin-1') as f:
                        f.write(line)
                    st.success("✅ Guardado!")
                    
                    numero_fijo = int(fij)
                    actualizados = actualizar_resultados_pendientes(numero_fijo, f_nueva, s_code)
                    
                    if actualizados > 0:
                        st.success(f"✅ {actualizados} predicción(es) actualizada(s)")
                    else:
                        st.warning(f"⚠️ No se encontraron predicciones pendientes")
                        st.info(f"💡 Verifica: Fecha={f_nueva.strftime('%Y-%m-%d')}, Sorteo={ses}")
                    
                    st.cache_data.clear()
                    st.session_state.invalid_dates_df = None
                    if os.path.exists(RUTA_CACHE):
                        os.remove(RUTA_CACHE)
                    time.sleep(1)
                    st.rerun()
                except Exception as err: st.error(f"❌ Error: {err}")

        with col_btn2:
            if st.button("⏪ Deshacer", use_container_width=True):
                try:
                    with open(RUTA_CSV, 'r', encoding='latin-1') as f: lines = f.readlines()
                    if len(lines) > 1:
                        lines.pop()
                        with open(RUTA_CSV, 'w', encoding='latin-1') as f: f.writelines(lines)
                        st.warning("🗑️ Eliminado")
                        st.cache_data.clear()
                        st.session_state.invalid_dates_df = None
                        if os.path.exists(RUTA_CACHE): os.remove(RUTA_CACHE)
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("⚠️ Vacío")
                except Exception as err: st.error(f"❌ Error: {err}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Últimos Resultados")
    
    fecha_ref_default = pd.Timestamp.now().normalize()
    target_sesion_default = "Tarde"
    info_ultimo_sorteo = None
    
    if not df_full.empty:
        sort_order_map = {'T': 0, 'N': 1}
        df_sort = df_full[df_full['Posicion'] == 'Fijo'].copy()
        if not df_sort.empty:
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
                        val_c = df_full.loc[mask & (df_full['Posicion'] == 'Centena'), 'Numero'].iloc[0] if (mask & (df_full['Posicion'] == 'Centena')).any() else 0
                        val_f = df_full.loc[mask & (df_full['Posicion'] == 'Fijo'), 'Numero'].iloc[0] if (mask & (df_full['Posicion'] == 'Fijo')).any() else 0
                        val_1 = df_full.loc[mask & (df_full['Posicion'] == '1er Corrido'), 'Numero'].iloc[0] if (mask & (df_full['Posicion'] == '1er Corrido')).any() else 0
                        val_2 = df_full.loc[mask & (df_full['Posicion'] == '2do Corrido'), 'Numero'].iloc[0] if (mask & (df_full['Posicion'] == '2do Corrido')).any() else 0
                        st.sidebar.markdown(f"Num: **{int(f'{val_c}{val_f:02d}')}**")
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

    st.sidebar.markdown("### 🎯 Configuración de Análisis")
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
        st.session_state.rerun_counter = 0
        if os.path.exists(RUTA_CACHE):
            os.remove(RUTA_CACHE)
        st.rerun()
    
    if modo_sorteo == "Tarde": 
        df_analisis = df_full[df_full['Tipo_Sorteo'] == 'T'].copy()
        nombre_sesion_backtest = "Tarde"
    elif modo_sorteo == "Noche": 
        df_analisis = df_full[df_full['Tipo_Sorteo'] == 'N'].copy()
        nombre_sesion_backtest = "Noche"
    else: 
        df_analisis = df_full.copy()
        nombre_sesion_backtest = "General (Todas)"
    
    if df_analisis.empty and modo_sorteo != "General":
        st.warning(f"⚠️ No hay datos para {modo_sorteo}. Cambia a 'General' o agrega sorteos.")
    
    if df_full.empty:
        st.warning("⚠️ No hay datos en el archivo CSV.")
        st.stop()

    orden_sesiones = {'Tarde': 0, 'Noche': 1}
    target_val = orden_sesiones[target_sesion]
    
    if target_val == 0: 
        df_backtest = df_analisis[df_analisis['Fecha'] < fecha_ref].copy()
    else: 
        df_backtest = df_analisis[(df_analisis['Fecha'] < fecha_ref) | ((df_analisis['Fecha'] == fecha_ref) & (df_analisis['Tipo_Sorteo'] == 'T'))].copy()

    mostrar_dashboard_efectividad()
    st.markdown("---")

    mostrar_tabla_personalidad_perfiles(obtener_historial_perfiles_cacheado(df_full, RUTA_CACHE))
    st.markdown("---")

    st.header("📜 Historial de Combinaciones y Estados")
    df_historial_perfiles_full = obtener_historial_perfiles_cacheado(df_full, RUTA_CACHE)
    
    if not df_historial_perfiles_full.empty:
        df_hist_view = df_historial_perfiles_full.copy()
        
        df_hist_view['ID_Unico'] = df_hist_view['Fecha'].astype(str).str[:10] + "_" + df_hist_view['Sorteo'].astype(str)
        df_hist_view = df_hist_view.drop_duplicates(subset=['ID_Unico'], keep='last').reset_index(drop=True)
        df_hist_view.drop(columns=['ID_Unico'], inplace=True)
        
        df_hist_view['Decena'] = df_hist_view['Numero'] // 10
        df_hist_view['Unidad'] = df_hist_view['Numero'] % 10
        
        sort_map = {'Noche': 1, 'Tarde': 0, 'N': 1, 'T': 0}
        df_hist_view['sort_key'] = df_hist_view['Sorteo'].astype(str).map(sort_map).fillna(0)
        df_hist_view = df_hist_view.sort_values(by=['Fecha', 'sort_key'], ascending=[False, False])
        
        df_hist_view['Fecha'] = pd.to_datetime(df_hist_view['Fecha']).dt.strftime('%d/%m/%Y')
        df_hist_view = df_hist_view.rename(columns={'Perfil': 'Estado Salida'})
        
        cols_display = ['Fecha', 'Sorteo', 'Numero', 'Decena', 'Unidad', 'Estado Salida']
        st.dataframe(df_hist_view[cols_display].head(30), hide_index=True, use_container_width=True)
        
        st.caption(f"✅ {len(df_hist_view)} sorteos únicos (mostrando los 30 más recientes)")
    else:
        st.warning("No se pudo generar el historial de perfiles.")

    st.markdown("---")

    df_oport_dec, df_oport_uni, gaps_dec, gaps_uni, freq_dec, freq_uni = analizar_oportunidad_por_digito(df_backtest, fecha_ref)
    
    if info_ultimo_sorteo:
        tipo_nombre = {'T': 'Tarde', 'N': 'Noche'}[info_ultimo_sorteo['tipo']]
        fecha_txt = info_ultimo_sorteo['fecha'].strftime('%d/%m/%Y')
        if tipo_nombre == 'Tarde':
            st.caption(f"✅ Estados calculados hasta **Tarde de {fecha_txt}** → Predice para **Noche**")
        else:
            st.caption(f"✅ Estados calculados hasta **Noche de {fecha_txt}** → Predice para **Tarde de mañana**")
    
    st.header(f"🎯 Estado de Dígitos ({target_sesion})")
    col1, col2 = st.columns(2)
    with col1: st.subheader("🔟 Decenas"); st.dataframe(df_oport_dec.sort_values('Punt. Base', ascending=False), hide_index=True)
    with col2: st.subheader("1️⃣ Unidades"); st.dataframe(df_oport_uni.sort_values('Punt. Base', ascending=False), hide_index=True)

    st.markdown("---")
    st.header("📅 Análisis de Perfiles (Motor Mejorado)")
    
    if st.button("🚀 Ejecutar Análisis", type="primary"):
        with st.spinner("Analizando..."):
            if not df_historial_perfiles_full.empty:
                distribuciones_cache = pre_calcular_distribuciones_perfiles(df_historial_perfiles_full)
                df_stats, transizioni, ultimo_perfil = analizar_estadisticas_perfiles(df_historial_perfiles_full, fecha_ref, distribuciones_cache=distribuciones_cache)
                estabilidad_digitos = calcular_estabilidad_historica_digitos(df_backtest)
                generar_sugerencia_fusionada(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles_full, fecha_ref, estabilidad_digitos, gaps_dec, gaps_uni, distribuciones_cache, freq_dec, freq_uni)
                
                top40 = obtener_prediccion_numeros_lista(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles_full, fecha_ref, estabilidad_digitos, gaps_dec, gaps_uni, distribuciones_cache, freq_dec, freq_uni)
                if len(top40) > 0:
                    score_prom = df_stats['Score'].mean() if 'Score' in df_stats.columns and not df_stats.empty else 0
                    perfil_top = df_stats.iloc[0]['Perfil'] if not df_stats.empty else 'N/A'
                    guardar_prediccion_en_historico(fecha_ref, target_sesion, top40, perfil_top, score_prom)
                    st.success("📝 Predicción guardada (Top 40)")
                
                st.markdown("---")
                st.subheader("📊 Estadística de Perfiles (Completa)")
                cols_tabla = ['Perfil', 'Frecuencia', 'Veces Normal', 'Veces Vencido', 'Veces Muy Vencido', 'Estado Actual', 'Estabilidad', 'Tiempo Limite', 'Alerta', 'Estado Ultima Salida', 'Estabilidad Ultima Salida', 'Exceso Ultima Salida']
                df_display = df_stats[cols_tabla].copy()
                st.dataframe(df_display.sort_values('Frecuencia', ascending=False), hide_index=True)

    st.markdown("---")
    st.header("🧪 Backtesting (Real)")
    st.info(f"📌 **Sesión para backtest:** {nombre_sesion_backtest}")
    
    sorteos_back = st.slider("Número de sorteos", 3, 15, 7, key="slider_backtest")
    
    if st.button("▶️ Iniciar Backtest"):
        if df_analisis.empty:
            st.error(f"❌ No hay datos para backtest de {nombre_sesion_backtest}")
        else:
            with st.spinner(f"🔄 Backtest para {nombre_sesion_backtest} ({sorteos_back} sorteos)..."):
                df_res, aciertos, total, debug_logs = ejecutar_backtest(df_analisis, sorteos_back, nombre_sesion_backtest, debug_mode=st.session_state.debug_mode)
                
                if total > 0:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Sorteos", total)
                    col2.metric("Aciertos", aciertos)
                    col3.metric("Efectividad", f"{round((aciertos/total)*100, 1)}%")
                    
                    if aciertos > 0: 
                        st.success(f"✨ {aciertos} de {total} aciertos en {nombre_sesion_backtest}")
                    else: 
                        st.warning(f"⚠️ 0 aciertos en {nombre_sesion_backtest}")
                    
                    if st.session_state.debug_mode and debug_logs:
                        st.markdown(f"### 🔍 Logging ({nombre_sesion_backtest})")
                        st.dataframe(pd.DataFrame(debug_logs), hide_index=True)
                    
                    with st.expander("📋 Ver resultados"):
                        st.dataframe(df_res, hide_index=True)

if __name__ == "__main__":
    main()