# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import os
import traceback
from collections import defaultdict, Counter
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN DE LA RUTA_RELATIVA ---
# Asegúrate de que el archivo Flotodo.csv esté dentro de una carpeta llamada 'Flotodo'
RUTA_CSV = 'Flotodo.csv' 

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Florida - Análisis de Sorteos",
    page_icon="🌴",
    layout="wide"
)

st.title("🌴 Florida - Análisis de Sorteos")
st.markdown("Sistema de Análisis para los sorteos de Florida (Tarde y Noche).")
st.info("ℹ️ **Importante:** Análisis especializado para 2 sorteos diarios (Tarde y Noche).")

# --- FUNCIÓN AUXILIAR PARA ELIMINAR ACENTOS ---
def remove_accents(input_str):
    if not isinstance(input_str, str):
        return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# --- FUNCIÓN PARA CARGAR Y PROCESAR DATOS ---
@st.cache_resource
def cargar_datos_flotodo(_ruta_csv, debug_mode=False):
    try:
        st.info("Cargando y procesando datos históricos de Florida...")
        ruta_csv_absoluta = _ruta_csv
        
        if not os.path.exists(ruta_csv_absoluta):
            st.error(f"❌ Error: No se encontró el archivo de datos de Florida.")
            st.error(f"La aplicación buscó el archivo en la ruta: {ruta_csv_absoluta}")
            st.warning("💡 **Solución:** Asegúrate de que el archivo 'Flotodo.csv' exista en la carpeta 'Flotodo'.")
            st.stop()
        
        with open(ruta_csv_absoluta, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        
        if not lines:
            st.error("El archivo CSV está vacío.")
            st.stop()
        
        header_line = lines[0].strip()
        column_names = header_line.split(';')
        
        data = []
        for line in lines[1:]:
            if line.strip():
                values = line.strip().split(';')
                # Aseguramos que haya al menos las columnas básicas
                if len(values) >= 5:
                    data.append(values)
        
        df_historial = pd.DataFrame(data, columns=column_names)
        
        if debug_mode:
            st.subheader("🔍 Examen de los Encabezados del CSV")
            st.write("**Lista completa de encabezados:**")
            st.code(header_line)
            st.dataframe(df_historial.head())
        
        # Renombramos columnas para estandarizar (asumimos estructura similar a Geotodo)
        # Si tu CSV tiene nombres diferentes, ajústalos aquí
        df_historial.rename(columns={
            'Fecha': 'Fecha',
            'Tarde/Noche': 'Tipo_Sorteo', # O como se llame en tu CSV
            'Fijo': 'Fijo',
            '1er Corrido': 'Primer_Corrido',
            '2do Corrido': 'Segundo_Corrido'
        }, inplace=True)
        
        df_historial['Fecha'] = pd.to_datetime(df_historial['Fecha'], dayfirst=True, errors='coerce')
        df_historial.dropna(subset=['Fecha'], inplace=True)
        
        st.write("Normalizando la columna 'Tipo_Sorteo' para Florida (T y N)...")
        # MAPEO AJUSTADO: Solo Tarde y Noche
        df_historial['Tipo_Sorteo'] = df_historial['Tipo_Sorteo'].astype(str).str.strip().str.upper().map({
            'TARDE': 'T', 'T': 'T',
            'NOCHE': 'N', 'N': 'N'
        }).fillna('OTRO')
        
        # Filtramos cualquier registro que no sea T o N (por seguridad)
        df_historial = df_historial[df_historial['Tipo_Sorteo'].isin(['T', 'N'])]
        
        st.success("Columna 'Tipo_Sorteo' normalizada (T, N).")
        if debug_mode:
            st.write("Valores únicos en 'Tipo_Sorteo' después de normalizar:", df_historial['Tipo_Sorteo'].unique())
        
        df_procesado = []
        for _, row in df_historial.iterrows():
            fecha = row['Fecha']
            tipo_sorteo = row['Tipo_Sorteo']
            # Aseguramos que los números sean leídos correctamente
            try:
                fijo = int(row['Fijo']) if pd.notna(row['Fijo']) else 0
                p1 = int(row['Primer_Corrido']) if pd.notna(row['Primer_Corrido']) else 0
                p2 = int(row['Segundo_Corrido']) if pd.notna(row['Segundo_Corrido']) else 0
            except ValueError:
                continue

            df_procesado.append({'Fecha': fecha, 'Tipo_Sorteo': tipo_sorteo, 'Numero': fijo, 'Posicion': 'Fijo'})
            df_procesado.append({'Fecha': fecha, 'Tipo_Sorteo': tipo_sorteo, 'Numero': p1, 'Posicion': '1er Corrido'})
            df_procesado.append({'Fecha': fecha, 'Tipo_Sorteo': tipo_sorteo, 'Numero': p2, 'Posicion': '2do Corrido'})
        
        df_historial = pd.DataFrame(df_procesado)
        df_historial['Numero'] = pd.to_numeric(df_historial['Numero'], errors='coerce')
        df_historial.dropna(subset=['Numero'], inplace=True)
        df_historial['Numero'] = df_historial['Numero'].astype(int)
        
        # ORDENAMIENTO AJUSTADO: Tarde (0) luego Noche (1)
        draw_order_map = {'T': 0, 'N': 1}
        df_historial['draw_order'] = df_historial['Tipo_Sorteo'].map(draw_order_map)
        df_historial['sort_key'] = df_historial['Fecha'] + pd.to_timedelta(df_historial['draw_order'], unit='h')
        df_historial = df_historial.sort_values(by='sort_key').reset_index(drop=True)
        df_historial.drop(columns=['draw_order', 'sort_key'], inplace=True)
        
        st.success("¡Datos de Florida cargados y procesados con éxito!")
        return df_historial
    except Exception as e:
        st.error(f"Error al cargar y procesar los datos de Florida: {str(e)}")
        if debug_mode:
            st.error(traceback.format_exc())
        st.stop()

# --- FUNCIÓN PARA CALCULAR ESTADO ACTUAL ---
def calcular_estado_actual(gap, promedio_gap):
    if pd.isna(promedio_gap) or promedio_gap == 0:
        return "Normal"
    if gap <= promedio_gap:
        return "Normal"
    elif gap > (promedio_gap * 1.5):
        return "Muy Vencido"
    else: 
        return "Vencido"

# --- FUNCIÓN PARA OBTENER ESTADO COMPLETO DE NÚMEROS ---
def get_full_state_dataframe(df_historial, fecha_referencia):
    st.info(f"📅 **Análisis de Estado:** Calculando el estado de todos los números hasta la fecha **{fecha_referencia.strftime('%d/%m/%Y')}**.")
    df_historial_filtrado = df_historial[df_historial['Fecha'] < fecha_referencia].copy()
    if df_historial_filtrado.empty:
        return pd.DataFrame(), {}

    df_maestro = pd.DataFrame({'Numero': range(100)})
    primera_fecha_historica = df_historial['Fecha'].min()
    st.info("Pre-calculando promedios históricos individuales para cada número...")
    
    historicos_numero = {}
    for i in range(100):
        fechas_i = df_historial_filtrado[df_historial_filtrado['Numero'] == i]['Fecha'].sort_values()
        gaps = fechas_i.diff().dt.days.dropna()
        if len(gaps) > 0:
            historicos_numero[i] = gaps.median()
        else:
            historicos_numero[i] = (fecha_referencia - primera_fecha_historica).days

    df_maestro['Decena'] = df_maestro['Numero'] // 10
    df_maestro['Unidad'] = df_maestro['Numero'] % 10
    
    ultima_aparicion_num_key = df_historial_filtrado.groupby('Numero')['Fecha'].max().reindex(range(100))
    ultima_aparicion_num_key.fillna(primera_fecha_historica, inplace=True)
    gap_num = (fecha_referencia - ultima_aparicion_num_key).dt.days
    df_maestro['Salto_Numero'] = gap_num
    df_maestro['Estado_Numero'] = df_maestro.apply(lambda row: calcular_estado_actual(row['Salto_Numero'], historicos_numero[row['Numero']]), axis=1)
    df_maestro['Última Aparición (Fecha)'] = ultima_aparicion_num_key.dt.strftime('%d/%m/%Y')
    frecuencia = df_historial_filtrado['Numero'].value_counts().reindex(range(100)).fillna(0)
    df_maestro['Total_Salidas_Historico'] = frecuencia

    return df_maestro, historicos_numero

# --- FUNCIÓN PARA CLASIFICAR NÚMEROS POR TEMPERATURA ---
def crear_mapa_de_calor_numeros(df_frecuencia, top_n=30, medio_n=30):
    df_ordenado = df_frecuencia.sort_values(by='Total_Salidas_Historico', ascending=False).reset_index(drop=True).copy()
    df_ordenado['Temperatura'] = '🧊 Frío'
    df_ordenado.loc[top_n : top_n + medio_n - 1, 'Temperatura'] = '🟡 Tibio'
    df_ordenado.loc[0 : top_n - 1, 'Temperatura'] = '🔥 Caliente'
    return df_ordenado

# --- FUNCIÓN PARA ANÁLISIS DE OPORTUNIDAD POR DÍGITO ---
def analizar_oportunidad_por_digito(df_historial, df_estados_completos, historicos_numero, modo_temperatura, fecha_inicio_rango, fecha_fin_rango, top_n_candidatos=5):
    st.info(f"🎯 **Análisis de Oportunidad por Dígito:** Iniciando análisis en modo: **{modo_temperatura}**.")
    
    if modo_temperatura == "Histórico Completo":
        df_temperatura = df_historial.copy()
        st.info(f"🌡️ **Análisis de Temperatura:** Se usará el **historial completo** de la sesión seleccionada.")
    else:
        end_of_day = fecha_fin_rango + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df_temperatura = df_historial[(df_historial['Fecha'] >= fecha_inicio_rango) & (df_historial['Fecha'] <= end_of_day)].copy()
        if df_temperatura.empty:
            st.warning("El rango seleccionado no contiene sorteos. Se usará el historial completo.")
            df_temperatura = df_historial.copy()
        else:
            st.success(f"✅ Análisis de Temperatura: Rango {fecha_inicio_rango.strftime('%d/%m/%Y')} a {fecha_fin_rango.strftime('%d/%m/%Y')}.")

    contador_decenas = Counter()
    contador_unidades = Counter()
    for num in df_temperatura['Numero']:
        contador_decenas[num // 10] += 1
        contador_unidades[num % 10] += 1

    df_frecuencia_decenas = pd.DataFrame.from_dict(contador_decenas, orient='index', columns=['Frecuencia Total']).reset_index()
    df_frecuencia_decenas.rename(columns={'index': 'Dígito'}, inplace=True)
    df_frecuencia_unidades = pd.DataFrame.from_dict(contador_unidades, orient='index', columns=['Frecuencia Total']).reset_index()
    df_frecuencia_unidades.rename(columns={'index': 'Dígito'}, inplace=True)

    df_frecuencia_decenas = df_frecuencia_decenas.sort_values(by='Frecuencia Total', ascending=False).reset_index(drop=True)
    df_frecuencia_decenas['Temperatura'] = '🟡 Tibio'
    if len(df_frecuencia_decenas) >= 3: df_frecuencia_decenas.loc[0:2, 'Temperatura'] = '🔥 Caliente'
    if len(df_frecuencia_decenas) >= 6: df_frecuencia_decenas.loc[3:5, 'Temperatura'] = '🟡 Tibio'
    if len(df_frecuencia_decenas) >= 7: df_frecuencia_decenas.loc[6:9, 'Temperatura'] = '🧊 Frío'
    
    df_frecuencia_unidades = df_frecuencia_unidades.sort_values(by='Frecuencia Total', ascending=False).reset_index(drop=True)
    df_frecuencia_unidades['Temperatura'] = '🟡 Tibio'
    if len(df_frecuencia_unidades) >= 3: df_frecuencia_unidades.loc[0:2, 'Temperatura'] = '🔥 Caliente'
    if len(df_frecuencia_unidades) >= 6: df_frecuencia_unidades.loc[3:5, 'Temperatura'] = '🟡 Tibio'
    if len(df_frecuencia_unidades) >= 7: df_frecuencia_unidades.loc[6:9, 'Temperatura'] = '🧊 Frío'
    
    mapa_temperatura_decenas = pd.Series(df_frecuencia_decenas.Temperatura.values, index=df_frecuencia_decenas.Dígito).to_dict()
    mapa_temperatura_unidades = pd.Series(df_frecuencia_unidades.Temperatura.values, index=df_frecuencia_unidades.Dígito).to_dict()
    
    puntuacion_temperatura_map = {'🔥 Caliente': 30, '🟡 Tibio': 20, '🧊 Frío': 10}
    resultados_decenas = []
    resultados_unidades = []
    
    for i in range(10):
        numeros_con_decena = df_estados_completos[df_estados_completos['Decena'] == i]
        numeros_con_unidad = df_estados_completos[df_estados_completos['Unidad'] == i]
        
        estado_decena = numeros_con_decena['Estado_Numero'].iloc[0] if not numeros_con_decena.empty else 'Normal'
        estado_unidad = numeros_con_unidad['Estado_Numero'].iloc[0] if not numeros_con_unidad.empty else 'Normal'

        puntuacion_base_decena = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}[estado_decena]
        puntuacion_base_unidad = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}[estado_unidad]

        if estado_decena == 'Normal':
            gaps_decena = [historicos_numero.get(num, 0) for num in range(100) if num // 10 == i]
            promedio = np.mean(gaps_decena) if gaps_decena else 1
            gaps_actuales = numeros_con_decena['Salto_Numero'].tolist()
            gap_actual = np.mean(gaps_actuales) if gaps_actuales else 0
            puntuacion_proactiva_decena = min(49, (gap_actual / promedio * 50) if promedio > 0 else 0)
        else:
            puntuacion_proactiva_decena = 0

        if estado_unidad == 'Normal':
            gaps_unidad = [historicos_numero.get(num, 0) for num in range(100) if num % 10 == i]
            promedio = np.mean(gaps_unidad) if gaps_unidad else 1
            gaps_actuales = numeros_con_unidad['Salto_Numero'].tolist()
            gap_actual = np.mean(gaps_actuales) if gaps_actuales else 0
            puntuacion_proactiva_unidad = min(49, (gap_actual / promedio * 50) if promedio > 0 else 0)
        else:
            puntuacion_proactiva_unidad = 0

        temperatura_decena = mapa_temperatura_decenas.get(i, '🟡 Tibio')
        temperatura_unidad = mapa_temperatura_unidades.get(i, '🟡 Tibio')
        puntuacion_temp_decena = puntuacion_temperatura_map.get(temperatura_decena, 20)
        puntuacion_temp_unidad = puntuacion_temperatura_map.get(temperatura_unidad, 20)

        puntuacion_total_decena = puntuacion_base_decena + puntuacion_proactiva_decena + puntuacion_temp_decena
        puntuacion_total_unidad = puntuacion_base_unidad + puntuacion_proactiva_unidad + puntuacion_temp_unidad

        resultados_decenas.append({'Dígito': i, 'Rol': 'Decena', 'Temperatura': temperatura_decena, 'Estado': estado_decena, 'Puntuación Base': puntuacion_base_decena, 'Puntuación Proactiva': round(puntuacion_proactiva_decena, 1), 'Puntuación Temperatura': puntuacion_temp_decena, 'Puntuación Total': round(puntuacion_total_decena, 1)})
        resultados_unidades.append({'Dígito': i, 'Rol': 'Unidad', 'Temperatura': temperatura_unidad, 'Estado': estado_unidad, 'Puntuación Base': puntuacion_base_unidad, 'Puntuación Proactiva': round(puntuacion_proactiva_unidad, 1), 'Puntuación Temperatura': puntuacion_temp_unidad, 'Puntuación Total': round(puntuacion_total_unidad, 1)})

    df_oportunidad_decenas = pd.DataFrame(resultados_decenas)
    df_oportunidad_unidades = pd.DataFrame(resultados_unidades)

    puntuacion_decena_map = df_oportunidad_decenas.set_index('Dígito')['Puntuación Total'].to_dict()
    puntuacion_unidad_map = df_oportunidad_unidades.set_index('Dígito')['Puntuación Total'].to_dict()

    candidatos = []
    for num in range(100):
        decena = num // 10
        unidad = num % 10
        score_total = puntuacion_decena_map.get(decena, 0) + puntuacion_unidad_map.get(unidad, 0)
        candidatos.append({'Numero': num, 'Puntuación Total': score_total})

    df_candidatos = pd.DataFrame(candidatos).sort_values(by='Puntuación Total', ascending=False).head(top_n_candidatos)
    df_candidatos['Numero'] = df_candidatos['Numero'].apply(lambda x: f"{x:02d}")

    return df_oportunidad_decenas, df_oportunidad_unidades, df_candidatos, mapa_temperatura_decenas, mapa_temperatura_unidades

# --- FUNCIÓN PARA MAPA DE CALOR POSICIONAL ---
def crear_mapa_calor_posicional(mapa_temp_decenas, mapa_temp_unidades):
    temp_valores = {'🔥 Caliente': 3, '🟡 Tibio': 2, '🧊 Frío': 1}
    matriz_calor = np.zeros((10, 10))
    for unidad in range(10):
        for decena in range(10):
            temp_dec = mapa_temp_decenas.get(decena, '🟡 Tibio')
            temp_uni = mapa_temp_unidades.get(unidad, '🟡 Tibio')
            puntuacion = temp_valores[temp_dec] * 10 + temp_valores[temp_uni]
            matriz_calor[unidad, decena] = puntuacion
    return pd.DataFrame(matriz_calor, index=range(10), columns=range(10))

def analizar_combinaciones_extremas(mapa_temp_decenas, mapa_temp_unidades, df_estados_completos):
    hot_cold, cold_hot, hot_hot, cold_cold = [], [], [], []
    puntuaciones_desequilibrio = {}
    columnas_requeridas = ['Decena', 'Unidad', 'Numero']
    for col in columnas_requeridas:
        if col not in df_estados_completos.columns:
            st.error(f"Falta la columna '{col}' en el DataFrame.")
            st.stop()
    
    for num in range(100):
        decena = num // 10
        unidad = num % 10
        temp_dec = mapa_temp_decenas.get(decena, '🟡 Tibio')
        temp_uni = mapa_temp_unidades.get(unidad, '🟡 Tibio')
        
        temp_valores = {'🔥 Caliente': 3, '🟡 Tibio': 2, '🧊 Frío': 1}
        score_dec = temp_valores[temp_dec]
        score_uni = temp_valores[temp_uni]
        puntuacion_desequilibrio = abs(score_dec - score_uni)
        puntuaciones_desequilibrio[num] = puntuacion_desequilibrio
        
        num_str = f"{num:02d}"
        if temp_dec == '🔥 Caliente' and temp_uni == '🧊 Frío': hot_cold.append(num_str)
        elif temp_dec == '🧊 Frío' and temp_uni == '🔥 Caliente': cold_hot.append(num_str)
        elif temp_dec == '🔥 Caliente' and temp_uni == '🔥 Caliente': hot_hot.append(num_str)
        elif temp_dec == '🧊 Frío' and temp_uni == '🧊 Frío': cold_cold.append(num_str)

    hot_cold.sort(key=lambda x: puntuaciones_desequilibrio[int(x)], reverse=True)
    cold_hot.sort(key=lambda x: puntuaciones_desequilibrio[int(x)], reverse=True)
    hot_hot.sort(key=lambda x: puntuaciones_desequilibrio[int(x)], reverse=True)
    cold_cold.sort(key=lambda x: puntuaciones_desequilibrio[int(x)], reverse=True)
    
    df_hot_cold = pd.DataFrame({'Número': hot_cold, 'Puntuación Desequilibrio': [puntuaciones_desequilibrio[int(x)] for x in hot_cold]})
    df_cold_hot = pd.DataFrame({'Número': cold_hot, 'Puntuación Desequilibrio': [puntuaciones_desequilibrio[int(x)] for x in cold_hot]})
    df_hot_hot = pd.DataFrame({'Número': hot_hot, 'Puntuación Desequilibrio': [puntuaciones_desequilibrio[int(x)] for x in hot_hot]})
    df_cold_cold = pd.DataFrame({'Número': cold_cold, 'Puntuación Desequilibrio': [puntuaciones_desequilibrio[int(x)] for x in cold_cold]})
    
    return df_hot_cold, df_cold_hot, df_hot_hot, df_cold_cold

# --- FUNCIÓN PARA AUDITORÍA HISTÓRICA ---
def generar_auditoria_doble_normal(df_historial):
    st.info("🔍 **Auditoría Histórica:** Buscando todos los eventos 'Doble Normal' en el historial para validar la aleatoriedad.")
    
    df_fijo = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    if df_fijo.empty:
        st.warning("No hay datos para auditoría.")
        return pd.DataFrame()

    fechas_unicas = df_fijo['Fecha'].unique()
    fechas_unicas = np.sort(fechas_unicas)
    
    auditoria = []
    
    for i, fecha in enumerate(fechas_unicas):
        df_hasta_ahora = df_fijo[df_fijo['Fecha'] < fecha].copy()
        sorteos_fecha = df_fijo[df_fijo['Fecha'] == fecha].sort_values(by='Tipo_Sorteo') # T, N
        
        for _, row in sorteos_fecha.iterrows():
            numero = row['Numero']
            decena = numero // 10
            unidad = numero % 10
            
            estado_dec = 'Normal'
            estado_uni = 'Normal'
            
            def calc_estado(num, df_hist):
                fechas = df_hist[df_hist['Numero'] == num]['Fecha'].sort_values()
                if len(fechas) == 0: return 'Normal', 0, 0
                gaps = fechas.diff().dt.days.dropna()
                if len(gaps) == 0: return 'Normal', 0, 0
                promedio = gaps.median()
                gap = (fecha - fechas.max()).days
                return calcular_estado_actual(gap, promedio), gap, promedio

            estado_dec, _, _ = calc_estado(decena, df_hasta_ahora)
            estado_uni, _, _ = calc_estado(unidad, df_hasta_ahora)
            
            if estado_dec == 'Normal' and estado_uni == 'Normal':
                df_auditoria_previa = pd.DataFrame(auditoria)
                
                if not df_auditoria_previa.empty and 'Fecha' in df_auditoria_previa.columns:
                    hits_previos = df_auditoria_previa[df_auditoria_previa['Fecha'] < fecha]
                    if not hits_previos.empty:
                        ultimo_hit = hits_previos.iloc[-1]
                        dias_pasados = (fecha - ultimo_hit['Fecha']).days
                    else:
                        dias_pasados = None
                else:
                    dias_pasados = None
                
                auditoria.append({
                    'Fecha': fecha,
                    'Sesión': row['Tipo_Sorteo'],
                    'Número': f"{numero:02d}",
                    'Decena': decena,
                    'Unidad': unidad,
                    'Días desde el anterior (Doble Normal)': dias_pasados
                })
    
    df_auditoria = pd.DataFrame(auditoria)
    if not df_auditoria.empty:
        df_auditoria['Fecha'] = df_auditoria['Fecha'].dt.strftime('%d/%m/%Y')
        df_auditoria = df_auditoria.sort_values(by='Fecha', ascending=False).reset_index(drop=True)
    
    return df_auditoria

# --- FUNCIÓN PARA BUSCAR PATRONES ---
def buscar_patrones_secuenciales(df_historial, max_longitud=3, nombre_sesion="General"):
    df_fijo = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    if df_fijo.empty:
        return {}
    
    secuencia = df_fijo['Numero'].tolist()
    if len(df_fijo) <= max_longitud:
        return {}
        
    patrones = {}
    for longitud in range(2, max_longitud + 1):
        for i in range(len(secuencia) - longitud):
            patron = tuple(secuencia[i:i+longitud])
            siguiente = secuencia[i+longitud] if i+longitud < len(secuencia) else None
            if siguiente is not None:
                if patron not in patrones: patrones[patron] = {}
                if siguiente not in patrones[patron]: patrones[patron][siguiente] = 0
                patrones[patron][siguiente] += 1
                
    patrones_ordenados = {}
    for patron, siguientes in patrones.items():
        siguientes_ordenados = sorted(siguientes.items(), key=lambda x: x[1], reverse=True)
        patrones_ordenados[patron] = siguientes_ordenados
    return patrones_ordenados

# --- FUNCIÓN PARA COMPORTAMIENTO DOBLE NORMAL ---
def analizar_comportamiento_doble_normal(df_historial):
    st.info("📊 **Análisis de Comportamiento del 'Doble Normal'**: Evaluando el estado completo de los números 'Doble Normal' a lo largo del historial.")
    df_fijo = df_historial[df_historial['Posicion'] == 'Fijo'].copy() 
    if df_fijo.empty: return pd.DataFrame()

    comportamiento = {'Normal': 0, 'Vencido': 0, 'Muy Vencido': 0}
    total_doble_normal = 0
    historial_fechas = {num: df_fijo[df_fijo['Numero'] == num]['Fecha'].sort_values() for num in range(100)}
    
    for i, row in df_fijo.iterrows():
        fecha_actual, numero_actual = row['Fecha'], row['Numero']
        decena, unidad = numero_actual // 10, numero_actual % 10
        es_doble_normal = True
        
        for digito in [decena, unidad]:
            fechas_digito = [f for f in historial_fechas[digito] if f < fecha_actual]
            if len(fechas_digito) == 0: es_doble_normal = False; break
            gaps = pd.Series(fechas_digito).diff().dt.days.dropna()
            if len(gaps) == 0: es_doble_normal = False; break
            promedio_gap = gaps.median()
            gap_actual = (fecha_actual - fechas_digito[-1]).days
            if calcular_estado_actual(gap_actual, promedio_gap) != 'Normal':
                es_doble_normal = False; break
        
        if es_doble_normal:
            total_doble_normal += 1
            fechas_numero = [f for f in historial_fechas[numero_actual] if f < fecha_actual]
            if not fechas_numero: continue
            gaps_num = pd.Series(fechas_numero).diff().dt.days.dropna()
            if len(gaps_num) == 0: continue
            promedio_gap_num = gaps_num.median()
            gap_actual_num = (fecha_actual - fechas_numero[-1]).days
            estado_num = calcular_estado_actual(gap_actual_num, promedio_gap_num)
            comportamiento[estado_num] += 1
            
    if total_doble_normal == 0:
        st.warning("No se encontraron números 'Doble Normal' en el historial.")
        return pd.DataFrame()

    df_comportamiento = pd.DataFrame(list(comportamiento.items()), columns=['Estado del Número', 'Cantidad'])
    df_comportamiento['Porcentaje'] = (df_comportamiento['Cantidad'] / total_doble_normal * 100).round(1)
    return df_comportamiento

# --- FUNCIÓN TABLA HISTÓRICO VISUAL ---
def crear_tabla_historico_visual_fijo(df_historial, num_ultimos=30):
    df_fijo = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    if df_fijo.empty:
        return pd.DataFrame()

    fecha_max = df_fijo['Fecha'].max()
    fecha_inicio_4s = fecha_max - pd.Timedelta(weeks=4)
    df_analisis_4s = df_fijo[df_fijo['Fecha'] >= fecha_inicio_4s].copy()
    
    if df_analisis_4s.empty:
        return pd.DataFrame()

    ultimas_fechas = df_fijo['Fecha'].unique()[-num_ultimos:]
    historial_visual = []
    
    # Ajuste: Solo iterar sobre T y N
    for fecha in sorted(ultimas_fechas, reverse=True):
        for sesion_key, sesion_nombre in [('N', 'Noche'), ('T', 'Tarde')]:
            resultado = df_fijo[(df_fijo['Fecha'] == fecha) & (df_fijo['Tipo_Sorteo'] == sesion_key)]
            if not resultado.empty:
                numero = resultado.iloc[0]['Numero']
                df_hasta_fecha = df_analisis_4s[df_analisis_4s['Fecha'] < fecha]
                
                def get_estado_en_fecha(num, df_hist, fecha_limite):
                    df_num = df_hist[df_hist['Numero'] == num]['Fecha'].sort_values()
                    if len(df_num) == 0: return 'Normal', 0, 0
                    gaps = df_num.diff().dt.days.dropna()
                    if len(gaps) == 0: return 'Normal', 0, 0
                    promedio = gaps.median() 
                    ultima_fecha = df_num.max()
                    gap = (fecha_limite - ultima_fecha).days
                    estado = calcular_estado_actual(gap, promedio)
                    return estado, gap, promedio
                
                estado_dec, _, _ = get_estado_en_fecha(numero, df_hasta_fecha, fecha)
                estado_uni, _, _ = get_estado_en_fecha(numero, df_hasta_fecha, fecha)
                es_doble_normal = (estado_dec == 'Normal' and estado_uni == 'Normal')
                
                contexto_hist = df_hasta_fecha if df_analisis_4s.empty else df_fijo.copy()
                df_num_hist = contexto_hist[contexto_hist['Numero'] == numero]
                
                if df_num_hist.empty:
                    temp = 'N/A'
                    estado_num = 'N/A'
                else:
                    df_freq_total = contexto_hist['Numero'].value_counts().reset_index()
                    df_freq_total.columns = ['Numero', 'Total_Salidas_Historico']
                    df_freq = crear_mapa_de_calor_numeros(df_freq_total)
                    row = df_freq[df_freq['Numero'] == numero]
                    if not row.empty:
                        temp = row['Temperatura'].iloc[0]
                    else:
                        temp = 'N/A'
                    
                    gaps_num = df_num_hist['Fecha'].diff().dt.days.dropna()
                    if len(gaps_num) == 0:
                        estado_num = 'N/A'
                    else:
                        promedio_gap_num = gaps_num.median()
                        ultima_fecha_num = df_num_hist['Fecha'].max()
                        gap_num = (fecha - ultima_fecha_num).days
                        estado_num = calcular_estado_actual(gap_num, promedio_gap_num)

                historial_visual.append({
                    'Fecha': fecha.strftime('%d/%m/%Y'),
                    'Sesión': sesion_nombre,
                    'Fijo': f"{numero:02d}",
                    'Es Doble Normal': es_doble_normal,
                    'Temperatura': temp,
                    'Estado': estado_num
                })

    return pd.DataFrame(historial_visual)

# --- FUNCIÓN ESTRATEGIA TENDENCIA ---
def generar_estrategia_tendencia(df_historial, fecha_referencia):
    df_fijo = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    df_fijo = df_fijo[df_fijo['Fecha'] < fecha_referencia].copy()
    if df_fijo.empty: return pd.DataFrame(), [], [], []

    estados_digito = {d: {'estado': 'Normal', 'gap': 0, 'promedio': 0} for d in range(10)}
    for d in range(10):
        numeros_con_digito = [n for n in range(100) if n // 10 == d or n % 10 == d]
        fechas_digito = df_fijo[df_fijo['Numero'].isin(numeros_con_digito)]['Fecha'].sort_values()
        gaps = fechas_digito.diff().dt.days.dropna()
        if len(gaps) > 0:
            promedio_gap = gaps.median()
            ultima_fecha = fechas_digito.max()
            gap_actual = (fecha_referencia - ultima_fecha).days
            estado = calcular_estado_actual(gap_actual, promedio_gap)
            estados_digito[d] = {'estado': estado, 'gap': gap_actual, 'promedio': promedio_gap}

    estados_numero = {n: {'estado': 'Normal', 'gap': 0, 'promedio': 0} for n in range(100)}
    for n in range(100):
        fechas_num = df_fijo[df_fijo['Numero'] == n]['Fecha'].sort_values()
        gaps = fechas_num.diff().dt.days.dropna()
        if len(gaps) > 0:
            promedio_gap = gaps.median()
            ultima_fecha_num = fechas_num.max()
            gap_actual = (fecha_referencia - ultima_fecha_num).days
            estado = calcular_estado_actual(gap_actual, promedio_gap)
            estados_numero[n] = {'estado': estado, 'gap': gap_actual, 'promedio': promedio_gap}

    # Matriz Completa
    todos_doble_normal = []
    for num in range(100):
        decena, unidad = num // 10, num % 10
        if (estados_digito[decena]['estado'] == 'Normal' and 
            estados_digito[unidad]['estado'] == 'Normal' and
            estados_numero[num]['estado'] == 'Normal'):
            todos_doble_normal.append(num)

    # Vencidos (Subset de la matriz)
    candidatos_vencidos = []
    for num in range(100):
        decena, unidad = num // 10, num % 10
        if (estados_digito[decena]['estado'] == 'Normal' and 
            estados_digito[unidad]['estado'] == 'Normal' and
            estados_numero[num]['estado'] in ['Vencido', 'Muy Vencido']):
            candidatos_vencidos.append(num)
    
    # Normales (Subset de la matriz)
    candidatos_normales = []
    for num in range(100):
        decena, unidad = num // 10, num % 10
        if (estados_digito[decena]['estado'] == 'Normal' and 
            estados_digito[unidad]['estado'] == 'Normal' and
            estados_numero[num]['estado'] == 'Normal'):
            
            fechas_num = df_fijo[df_fijo['Numero'] == num]['Fecha'].sort_values().tolist()
            if len(fechas_num) < 56: continue
                
            gaps_semanales = []
            for i in range(len(fechas_num) - 1, 0, -7):
                if i - 7 < 0: break
                gap_semanal = (fechas_num[i] - fechas_num[i-7]).days
                if gap_semanal > 0:
                    gaps_semanales.append(gap_semanal)
            
            if gaps_semanales:
                gap_promedio_semanal = np.mean(gaps_semanales)
                gap_actual_semanal = (fecha_referencia - fechas_num[-1]).days
                if gap_actual_semanal < gap_promedio_semanal:
                    candidatos_normales.append((num, gap_actual_semanal, gap_promedio_semanal))
    
    candidatos_normales.sort(key=lambda x: x[1])
    candidatos_normales = [num for num, _, _ in candidatos_normales[:10]]
    
    df_completa = pd.DataFrame({'Número': [f"{n:02d}" for n in todos_doble_normal]})
    
    candidatos_finales = sorted(list(set(candidatos_vencidos + candidatos_normales)))
    
    return df_completa, candidatos_finales, sorted(candidatos_vencidos), sorted(candidatos_normales)

# --- FUNCIÓN PRINCIPAL ---
def main():
    st.sidebar.header("⚙️ Opciones de Análisis - Florida")
    
    debug_mode = st.sidebar.checkbox("🔍 Activar Modo Diagnóstico (CSV)", value=False)
    
    st.sidebar.subheader("📊 Modo de Análisis de Datos")
    # OPCIONES AJUSTADAS: Solo Tarde y Noche
    modo_sorteo = st.sidebar.radio(
        "Selecciona el conjunto de datos a analizar:",
        ["Análisis General (Todos los sorteos)", "Análisis por Sesión: Tarde (T)", "Análisis por Sesión: Noche (N)"]
    )
    
    modo_analisis = st.sidebar.radio(
        "Modo de Análisis Principal:",
        ["Análisis Actual (usando fecha de hoy)", "Análisis Personalizado"]
    )

    if modo_analisis == "Análisis Personalizado":
        fecha_referencia = st.sidebar.date_input("Selecciona la fecha de referencia:", value=datetime.now().date(), format="DD/MM/YYYY")
        fecha_referencia = pd.to_datetime(fecha_referencia).tz_localize(None)
    else:
        fecha_referencia = pd.Timestamp.now(tz=None)
        st.sidebar.info(f"Analizando con la fecha de hoy: {fecha_referencia.strftime('%d/%m/%Y')}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌡️ Modo de Temperatura de Dígitos (Almanaque)")
    modo_temperatura = st.sidebar.radio(
        "Selecciona modo para calcular la temperatura:",
        ["Histórico Completo", "Personalizado por Rango"]
    )
    
    fecha_inicio_rango, fecha_fin_rango = None, None
    if modo_temperatura == "Personalizado por Rango":
        st.sidebar.markdown("**Selecciona el rango de fechas:**")
        fecha_inicio_rango = st.sidebar.date_input("Fecha de Inicio:", value=fecha_referencia - pd.Timedelta(days=30), format="DD/MM/YYYY")
        fecha_fin_rango = st.sidebar.date_input("Fecha de Fin:", value=fecha_referencia - pd.Timedelta(days=1), format="DD/MM/YYYY")
        if fecha_inicio_rango > fecha_fin_rango:
            st.sidebar.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏆️ Análisis de Top Números")
    top_n_candidatos = st.slider("Top N de Números Candidatos a mostrar:", min_value=1, max_value=20, value=5, step=1)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Forzar Recarga de Datos"):
        st.cache_resource.clear()
        st.sidebar.success("¡Cache limpio! Recargando...")
        st.rerun()

    # Llamada a función de carga específica para Florida
    df_historial_completo = cargar_datos_flotodo(RUTA_CSV, debug_mode)

    if df_historial_completo is not None:
        if "Tarde" in modo_sorteo:
            df_analisis = df_historial_completo[df_historial_completo['Tipo_Sorteo'] == 'T'].copy()
            titulo_app = f"Análisis de la Sesión: Tarde (T)"
        elif "Noche" in modo_sorteo:
            df_analisis = df_historial_completo[df_historial_completo['Tipo_Sorteo'] == 'N'].copy()
            titulo_app = f"Análisis de la Sesión: Noche (N)"
        else: 
            df_analisis = df_historial_completo.copy()
            titulo_app = "Análisis General de Sorteos"
        
        st.title(f"🌴 {titulo_app}")
        
        if df_analisis.empty:
            st.warning(f"No hay datos para la sesión seleccionada ({modo_sorteo}).")
            st.stop()

        st.sidebar.markdown("---")
        # INFO SIDEBAR AJUSTADA: Eliminada Mañana
        if 'T' in df_analisis['Tipo_Sorteo'].unique():
            ultimo_sorteo_T = df_analisis[df_analisis['Tipo_Sorteo'] == 'T'].iloc[-1]
            st.sidebar.info(f"Último sorteo **Tarde**: {ultimo_sorteo_T['Fecha'].strftime('%d/%m/%Y')} (Fijo: {ultimo_sorteo_T['Numero']})")
        if 'N' in df_analisis['Tipo_Sorteo'].unique():
            ultimo_sorteo_N = df_analisis[df_analisis['Tipo_Sorteo'] == 'N'].iloc[-1]
            st.sidebar.info(f"Último sorteo **Noche**: {ultimo_sorteo_N['Fecha'].strftime('%d/%m/%Y')} (Fijo: {ultimo_sorteo_N['Numero']})")
        
        frecuencia_numeros_historica_general = df_historial_completo['Numero'].value_counts().reset_index()
        frecuencia_numeros_historica_general.columns = ['Numero', 'Total_Salidas_Historico']
        df_clasificacion_general = crear_mapa_de_calor_numeros(frecuencia_numeros_historica_general)
        
        df_estados_completos, historicos_numero = get_full_state_dataframe(df_analisis, fecha_referencia)
        
        if df_estados_completos.empty:
            st.error("No se pudo calcular el estado de los números para la fecha de referencia.")
            st.stop()

        frecuencia_numeros_historica = df_analisis['Numero'].value_counts().reset_index()
        frecuencia_numeros_historica.columns = ['Numero', 'Total_Salidas_Historico']
        df_clasificacion_actual = crear_mapa_de_calor_numeros(frecuencia_numeros_historica)
        
        fecha_inicio_rango_safe = pd.to_datetime(fecha_inicio_rango).tz_localize(None) if fecha_inicio_rango else None
        fecha_fin_rango_safe = pd.to_datetime(fecha_fin_rango).tz_localize(None) if fecha_fin_rango else None
        
        df_oportunidad_decenas, df_oportunidad_unidades, top_candidatos, mapa_temp_decenas, mapa_temp_unidades = analizar_oportunidad_por_digito(
            df_analisis, df_estados_completos, historicos_numero, 
            modo_temperatura, fecha_inicio_rango_safe, fecha_fin_rango_safe,
            top_n_candidatos
        )
        
        # --- SECCIÓN 1: NÚMEROS CON OPORTUNIDAD ---
        st.markdown("---")
        st.header("🎯 Números con Oportunidad (Debidos) por Grupo")
        st.markdown("Intersección de los números de cada grupo (Calientes, Tibios, Fríos) con los que están en estado 'Vencido' o 'Muy Vencido'.")

        oportunidades_por_grupo = {}
        grupos_analizar = ['🔥 Caliente', '🟡 Tibio', '🧊 Frío']
        for temp in grupos_analizar:
            numeros_grupo_df = df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == temp]
            con_estado = numeros_grupo_df.merge(df_estados_completos[['Numero', 'Estado_Numero']], on='Numero')
            con_oportunidad = con_estado[con_estado['Estado_Numero'].isin(['Vencido', 'Muy Vencido'])]
            oportunidades_por_grupo[temp] = con_oportunidad
        
        tabs = st.tabs(grupos_analizar)
        for i, temp in enumerate(grupos_analizar):
            with tabs[i]:
                df_oportunidad_grupo = oportunidades_por_grupo[temp]
                st.subheader(f"Análisis del Grupo {temp}")
                if df_oportunidad_grupo.empty:
                    st.warning(f"Actualmente, ninguno de los números del grupo '{temp}' se encuentra en estado de 'Oportunidad' hasta la fecha {fecha_referencia.strftime('%d/%m/%Y')}.")
                else:
                    st.success(f"Se encontraron {len(df_oportunidad_grupo)} números con 'Oportunidad' en el grupo '{temp}'.")
                    st.dataframe(df_oportunidad_grupo[['Numero', 'Total_Salidas_Historico', 'Estado_Numero']], width='stretch', hide_index=True)

        # --- SECCIÓN 2: CLASIFICACIÓN GENERAL ---
        st.markdown("---")
        st.header(f"🌡️ Clasificación de Números (Basada en {modo_sorteo})")
        
        col_cal, col_tib, col_fri = st.columns(3)
        with col_cal:
            st.metric("🔥 Calientes (Top 30)", f"{len(df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == '🔥 Caliente'])} números")
            calientes_lista = df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == '🔥 Caliente']['Numero'].tolist()
            st.write(", ".join(map(str, calientes_lista)))
        with col_tib:
            st.metric("🟡 Tibios (Siguientes 30)", f"{len(df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == '🟡 Tibio'])} números")
            tibios_lista = df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == '🟡 Tibio']['Numero'].tolist()
            st.write(", ".join(map(str, tibios_lista)))
        with col_fri:
            st.metric("🧊 Fríos (Últimos 40)", f"{len(df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == '🧊 Frío'])} números")
            frios_lista = df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == '🧊 Frío']['Numero'].tolist()
            st.write(", ".join(map(str, frios_lista)))
        
        # --- SECCIÓN 3: ESTADO DE LOS GRUPOS ---
        st.markdown("---")
        st.header("📊 Estado de los Grupos por Temperatura (Rendimiento Reciente)")
        st.info(f"📅 **Período de Análisis:** Se compara el rendimiento de los últimos **7 días** (hasta el {fecha_referencia.strftime('%d/%m/%Y')}) con el promedio histórico.")

        def calcular_estado_grupo_corregido(df_historial, df_clasificacion_actual, fecha_referencia, dias_recientes=7):
            resultados = {}
            df_historial_filtrado = df_historial[df_historial['Fecha'] < fecha_referencia].copy()
            total_dias_historia = (df_historial_filtrado['Fecha'].max() - df_historial_filtrado['Fecha'].min()).days + 1
            if total_dias_historia == 0: return {}

            for temp in ['🔥 Caliente', '🟡 Tibio', '🧊 Frío']:
                numeros_grupo = df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == temp]['Numero'].tolist()
                if not numeros_grupo: continue
                    
                apariciones_historicas_grupo = df_historial_filtrado[df_historial_filtrado['Numero'].isin(numeros_grupo)]
                frecuencia_historica = len(apariciones_historicas_grupo) / total_dias_historia

                fecha_inicio_reciente = fecha_referencia - pd.Timedelta(days=dias_recientes)
                df_reciente = df_historial_filtrado[df_historial_filtrado['Fecha'] >= fecha_inicio_reciente]
                apariciones_recientes_grupo = df_reciente[df_reciente['Numero'].isin(numeros_grupo)]
                frecuencia_reciente = len(apariciones_recientes_grupo) / dias_recientes

                ratio_rendimiento = frecuencia_reciente / frecuencia_historica if frecuencia_historica > 0 else 0
                
                if ratio_rendimiento < 0.7: estado_grupo = "Vencido (Enfriado)"
                elif ratio_rendimiento > 1.3: estado_grupo = "Sobrecalentado"
                else: estado_grupo = "Normal"
                    
                resultados[temp] = {
                    'Estado': estado_grupo, 'Frecuencia Histórica (por día)': round(frecuencia_historica, 2),
                    'Frecuencia Reciente (por día)': round(frecuencia_reciente, 2),
                    'Ratio Rendimiento (%)': round(ratio_rendimiento * 100, 1),
                    'Total Apariciones (7 días)': len(apariciones_recientes_grupo), 'Total Números': len(numeros_grupo)
                }
            return resultados

        grupos_estado_corregido = calcular_estado_grupo_corregido(df_analisis, df_clasificacion_actual, fecha_referencia)
        
        # --- SECCIÓN 3.5: SUGERENCIA ESTRATÉGICA ---
        st.markdown("---")
        st.header("💡 Sugerencia Estratégica Basada en Tendencias")
        
        sugerencia_grupo = None
        sugerencia_texto = ""
        numeros_sugeridos = []

        if grupos_estado_corregido:
            df_grupos_estado = pd.DataFrame.from_dict(grupos_estado_corregido, orient='index').reset_index()
            df_grupos_estado.rename(columns={'index': 'Grupo'}, inplace=True)
            
            grupo_vencido = df_grupos_estado[df_grupos_estado['Estado'] == 'Vencido (Enfriado)']
            
            if not grupo_vencido.empty:
                grupo_vencido = grupo_vencido.sort_values(by='Ratio Rendimiento (%)').iloc[0]
                sugerencia_grupo = grupo_vencido['Grupo']
                sugerencia_texto = f"El grupo **{sugerencia_grupo}** está 'Vencido (Enfriado)'."
            else:
                grupo_normal = df_grupos_estado[df_grupos_estado['Estado'] == 'Normal']
                if not grupo_normal.empty:
                    grupo_normal = grupo_normal.sort_values(by='Ratio Rendimiento (%)').iloc[0]
                    sugerencia_grupo = grupo_normal['Grupo']
                    sugerencia_texto = f"No hay grupos 'Vencidos'. El grupo **{sugerencia_grupo}** tiene el rendimiento más bajo ('Normal')."
                else: 
                    sugerencia_texto = "Todos los grupos muestran rendimiento 'Normal' o 'Sobrecalentado'."
            
            if sugerencia_grupo:
                st.info(sugerencia_texto)
                numeros_grupo_sugerido = df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == sugerencia_grupo]['Numero'].tolist()
                numeros_debidos_grupo = df_estados_completos[df_estados_completos['Numero'].isin(numeros_grupo_sugerido)]
                numeros_debidos_grupo = numeros_debidos_grupo[numeros_debidos_grupo['Estado_Numero'].isin(['Vencido', 'Muy Vencido'])]
                
                if not numeros_debidos_grupo.empty:
                    numeros_debidos_grupo = numeros_debidos_grupo.sort_values(by='Salto_Numero', ascending=False)
                    numeros_sugeridos = numeros_debidos_grupo.head(5)['Numero'].apply(lambda x: f"{x:02d}").tolist()
                    st.success(f"**Números más prometedores del grupo {sugerencia_grupo}:** {', '.join(numeros_sugeridos)}")
                else: 
                    st.warning(f"El grupo {sugerencia_grupo} es una oportunidad, pero no contiene números individualmente 'Vencidos' o 'Muy Vencidos' en este momento.")
            else: 
                st.warning(sugerencia_texto)

            st.dataframe(df_grupos_estado, width='stretch', hide_index=True)
        else: 
            st.warning("No se pudo calcular el estado de los grupos.")
        
        # --- SECCIÓN 4: ANÁLISIS DE OPORTUNIDAD POR DÍGITO ---
        st.markdown("---")
        st.header("🎯 Análisis de Oportunidad por Dígito (Decenas y Unidades)")
        col_dec, col_uni = st.columns(2)
        with col_dec:
            st.subheader("📊 Oportunidad por Decena")
            st.dataframe(df_oportunidad_decenas.sort_values(by='Puntuación Total', ascending=False), width='stretch', hide_index=True)
        with col_uni:
            st.subheader("📊 Oportunidad por Unidad")
            st.dataframe(df_oportunidad_unidades.sort_values(by='Puntuación Total', ascending=False), width='stretch', hide_index=True)

        st.markdown("---")
        st.subheader(f"🏆 Top {top_n_candidatos} Números Candidatos (Puntuación Combinada)")
        st.dataframe(top_candidatos, width='stretch', hide_index=True)
        
        # --- SECCIÓN 5: MAPA DE CALOR POSICIONAL ---
        st.markdown("---")
        st.header("🗺️ Mapa de Calor Posicional")
        st.info(f"ℹ️ **Contexto del Análisis:** Este mapa se basa en la temperatura de los dígitos calculada según el modo seleccionado ('{modo_temperatura}').")
        
        df_mapa_calor = crear_mapa_calor_posicional(mapa_temp_decenas, mapa_temp_unidades)
        df_hot_cold, df_cold_hot, df_hot_hot, df_cold_cold = analizar_combinaciones_extremas(mapa_temp_decenas, mapa_temp_unidades, df_estados_completos)
        
        st.subheader("Mapa de Calor de Combinaciones Posicionales")
        st.markdown("**Cómo leer este mapa:** - **Eje X:** Dígito de la Decena (0-9) - **Eje Y:** Dígito de la Unidad (0-9) - **Colores:** Representan la combinación de temperatura.")
        
        cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_mapa_calor, annot=True, fmt=".0f", cmap=cmap, center=22, xticklabels=range(10), yticklabels=range(10), cbar_kws={'label': 'Puntuación de Temperatura Combinada'})
        ax.set_xlabel('Dígito de la Decena'); ax.set_ylabel('Dígito de la Unidad'); ax.set_title('Mapa de Calor Posicional')
        st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("📊 Análisis de Combinaciones Extremas")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔥🧊 Caliente-Frío (Máxima Oportunidad)"); st.markdown("Números donde la decena está muy caliente y la unidad muy fría.")
            if not df_hot_cold.empty: st.dataframe(df_hot_cold, width='stretch', hide_index=True)
            else: st.warning("No hay combinaciones Caliente-Frío.")
            st.markdown("---"); st.subheader("🔥🔥 Doble Caliente (Alta Probabilidad)"); st.markdown("Números donde ambas posiciones están muy calientes.")
            if not df_hot_hot.empty: st.dataframe(df_hot_hot, width='stretch', hide_index=True)
            else: st.warning("No hay combinaciones Doble Caliente.")
        with col2:
            st.subheader("🧊🔥 Frío-Caliente (Segunda Oportunidad)"); st.markdown("Números donde la unidad está muy caliente y la decena muy fría.")
            if not df_cold_hot.empty: st.dataframe(df_cold_hot, width='stretch', hide_index=True)
            else: st.warning("No hay combinaciones Frío-Caliente.")
            st.markdown("---"); st.subheader("🧊🧊 Doble Frío (Posible Sorpresa)"); st.markdown("Números donde ambas posiciones están muy frías.")
            if not df_cold_cold.empty: st.dataframe(df_cold_cold, width='stretch', hide_index=True)
            else: st.write("No hay combinaciones Doble Frío.")

        # --- SECCIÓN 6: PATRONES SECUENCIALES ---
        st.markdown("---")
        st.header("🔍 Búsqueda de Patrones Secuenciales")
        st.markdown("Busca patrones en la columna Fijo.")
        df_analisis_para_patrones = df_analisis[df_analisis['Fecha'] < fecha_referencia].copy()
        nombre_sesion_para_patrones = modo_sorteo.split(':')[-1].strip()
        patrones = buscar_patrones_secuenciales(df_analisis_para_patrones, max_longitud=3, nombre_sesion=nombre_sesion_para_patrones)
        
        if patrones:
            st.subheader("Últimos Patrones Detectados y Posibles Siguientes")
            df_fijo = df_analisis_para_patrones[df_analisis_para_patrones['Posicion'] == 'Fijo'].copy()
            ultimos_numeros = df_fijo.tail(3)['Numero'].tolist()
            
            if len(ultimos_numeros) >= 2:
                patron_2 = tuple(ultimos_numeros[-2:])
                st.write(f"**Último patrón de 2 números:** {patron_2[0]} → {patron_2[1]}")
                if patron_2 in patrones:
                    siguientes_2 = patrones[patron_2][:3]
                    df_siguientes_2 = pd.DataFrame(siguientes_2, columns=['Siguiente Número', 'Frecuencia'])
                    st.dataframe(df_siguientes_2, width='stretch', hide_index=True)
                    if siguientes_2:
                        recomendacion_2 = siguientes_2[0][0]
                        st.success(f"Próximo probable: **{recomendacion_2:02d}**")
                else:
                    st.warning(f"El patrón reciente `{patron_2[0]} → {patron_2[1]}` no se ha repetido.")
            
            if len(ultimos_numeros) >= 3:
                patron_3 = tuple(ultimos_numeros[-3:])
                st.write(f"**Último patrón de 3 números:** {patron_3[0]} → {patron_3[1]} → {patron_3[2]}")
                if patron_3 in patrones:
                    siguientes_3 = patrones[patron_3][:3]
                    df_siguientes_3 = pd.DataFrame(siguientes_3, columns=['Siguiente Número', 'Frecuencia'])
                    st.dataframe(df_siguientes_3, width='stretch', hide_index=True)
                    if siguientes_3:
                        recomendacion_3 = siguientes_3[0][0]
                        st.success(f"Próximo probable: **{recomendacion_3:02d}**")
                else:
                    st.warning(f"El patrón reciente `{patron_3[0]} → {patron_3[1]} → {patron_3[2]}` no se ha repetido.")
            
            st.markdown("---"); st.subheader("Patrones Más Frecuentes")
            frecuencia_patrones = {pat: sum(f for _, f in sigs) for pat, sigs in patrones.items()}
            patrones_ordenados = sorted(frecuencia_patrones.items(), key=lambda x: x[1], reverse=True)
            top_patrones = patrones_ordenados[:10]
            df_top_patrones = pd.DataFrame(top_patrones, columns=['Patrón', 'Frecuencia Total'])
            df_top_patrones['Patrón'] = df_top_patrones['Patrón'].apply(lambda x: ' → '.join([str(n) for n in x]))
            st.dataframe(df_top_patrones, width='stretch', hide_index=True)
        else: 
            st.warning("No se encontraron patrones.")

        # --- NUEVA SECCIÓN: AUDITORÍA HISTÓRICA (CORROBORACIÓN) ---
        st.markdown("---")
        st.header("🔍 Auditoría Histórica de 'Doble Normal' (Corroboración de Aciertos)")
        st.markdown("""
        Para responder a tu duda sobre **'¿Por qué tantos aciertos?'**, esta sección muestra la historia completa de cada vez que salió un 'Doble Normal'.
        
        *   **Días desde el anterior:** Muestra la cantidad de sorteos transcurridos desde la última vez que salió un 'Doble Normal'.
        *   **Histograma:** Gráfico que muestra qué tan frecuente es que salgan (1 día, 2 días, etc.).
        
        Si la lotería es aleatoria, verás una variedad de días (a veces 1, a veces 10).
        """)
        
        df_auditoria = generar_auditoria_doble_normal(df_analisis)
        
        if not df_auditoria.empty:
            # Tabla de Auditoría
            st.dataframe(df_auditoria, width='stretch', hide_index=True)
            
            # Histograma de "Días entre aciertos"
            st.subheader("📊 Distribución de Tiempo (Días entre aciertos Doble Normal)")
            gaps_auditoria = df_auditoria['Días desde el anterior (Doble Normal)'].dropna()
            
            if not gaps_auditoria.empty:
                fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
                sns.histplot(gaps_auditoria, bins=range(1, 21), kde=True, ax=ax_hist, color='skyblue')
                ax_hist.set_title("Distribución de 'Días desde el anterior'")
                ax_hist.set_xlabel("Días desde el último acierto")
                ax_hist.set_ylabel("Frecuencia")
                st.pyplot(fig_hist)
                
                st.write("""
                **Análisis:**
                - Observa el gráfico. 
                - Si ves muchas barras a la izquierda (1, 2, 3 días), significa que los 'Doble Normal' suelen salir seguidos.
                - Si ves barras dispersas (ej. 1, 5, 2, 9 días), confirma que es **aleatorio y cambiante**, como sospechabas.
                - La barra más alta indica el periodo de tiempo más común entre apariciones.
                """)
            else:
                st.info("No hay suficientes datos para calcular el tiempo entre aciertos.")
        
        # --- SECCIÓN 7: COMPORTAMIENTO DOBLE NORMAL ---
        st.markdown("---")
        st.header("📊 Análisis de Comportamiento del 'Doble Normal'")
        df_comportamiento = analizar_comportamiento_doble_normal(df_analisis)
        if not df_comportamiento.empty:
            st.dataframe(df_comportamiento, width='stretch', hide_index=True)
        
        # --- SECCIÓN 8: MATRIZ COMPLETA Y CANDIDATOS ---
        st.markdown("---")
        st.header("🧠 Generador y Evaluador de Estrategia de Tendencia")
        st.markdown("Estrategia Mejorada: Números 'Doble Normal' que están 'Vencidos' o 'Muy Vencidos', o 'Normal-Normal' con buen rendimiento semanal.")
        
        st.subheader(f"Candidatos para el {fecha_referencia.strftime('%d/%m/%Y')}")
        
        df_completa, candidatos_finales, candidatos_vencidos, candidatos_normales = generar_estrategia_tendencia(df_analisis, fecha_referencia)
        
        # --- NUEVA SUB-SECCIÓN: MATRIZ COMPLETA ---
        st.markdown("---")
        st.subheader("🔵 Matriz Completa: Todos los números 'Doble Normal'")
        st.markdown("A continuación se muestran **todos** los números que cumplen con la condición 'Doble Normal' (Decena Normal + Unidad Normal) para la fecha de hoy, mostrando su estado individual.")
        
        if not df_completa.empty:
            st.dataframe(df_completa, use_container_width=True)
        else:
            st.info("No hay números 'Doble Normal' para hoy.")
        
        st.markdown("---")
        if candidatos_vencidos:
            st.warning(f"🔥 **Candidatos Urgentes (Doble Normal Vencidos)**: {len(candidatos_vencidos)} números.")
            st.write(", ".join([f"{n:02d}" for n in candidatos_vencidos]))
        else:
            st.info("No hay candidatos 'Doble Normal Vencidos' para hoy.")
            
        if candidatos_normales:
            st.success(f"🧊 **Candidatos Estables (Doble Normal Rendimiento Semanal)**: {len(candidatos_normales)} números.")
            st.write(", ".join([f"{n:02d}" for n in candidatos_normales]))
        else:
            st.info("No hay candidatos 'Doble Normal Estables' para hoy.")

        if candidatos_finales:
            st.markdown("---")
            st.info(f"📋 **Lista Unificada ({len(candidatos_finales)} números):**")
            st.write(", ".join([f"{n:02d}" for n in candidatos_finales]))
        else:
            st.warning("La estrategia no encuentra candidatos que cumplan los criterios para hoy.")

        # --- SECCIÓN 9: HISTORIAL VISUAL ENRIQUECIDA ---
        st.markdown("---")
        st.header("📊 Historial Visual Enriquecido (Últimos 30 Fijos)")
        st.markdown("Tabla de los últimos 30 resultados 'Fijo'. Importante: El estado 'Doble Normal' se calcula usando datos de las últimas 4 semanas.")
        
        df_historico_visual = crear_tabla_historico_visual_fijo(df_analisis)
        if not df_historico_visual.empty:
            total_sorteos = len(df_historico_visual)
            aciertos = df_historico_visual['Es Doble Normal'].sum()
            tasa_acierto = (aciertos / total_sorteos * 100) if total_sorteos > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de sorteos analizados", total_sorteos)
            col2.metric("Aciertos 'Doble Normal'", aciertos)
            col3.metric("Tasa de Acierto", f"{tasa_acierto:.1f}%")
            
            def resaltar_fila_doble_normal(row):
                return ['background-color: #d4edda' if row['Es Doble Normal'] else '' for _ in row]
            
            st.dataframe(df_historico_visual.style.apply(resaltar_fila_doble_normal, axis=1), width='stretch', hide_index=True)

        # --- SECCIÓN 10: COMPARACIÓN DE SESIONES ---
        if modo_sorteo != "Análisis General (Todos los sorteos)":
            st.markdown("---")
            st.header("🔍 Comparación de Sesiones")
            st.markdown(f"Comparación de los Top 10 números de la sesión actual ({modo_sorteo.split(':')[-1].strip()}) con su rendimiento en las otras sesiones y en el análisis general.")

            top_10_actual = df_clasificacion_actual.head(10)
            numeros_top_10 = top_10_actual['Numero'].tolist()

            # AJUSTE: Comparar solo con T y N
            sesion_actual_key = modo_sorteo.split(':')[-1].strip()
            otras_sesiones = [k for k in ['T', 'N'] if k != sesion_actual_key]
            
            datos_comparacion = []
            for num in numeros_top_10:
                fila = {'Número': f"{num:02d}"}
                freq_actual = df_analisis[df_analisis['Numero'] == num].shape[0]
                fila[f'Frecuencia en {modo_sorteo.split(":")[-1].strip()}'] = freq_actual
                for key in otras_sesiones:
                    df_otra_sesion = df_historial_completo[df_historial_completo['Tipo_Sorteo'] == key]
                    freq_otra = df_otra_sesion[df_otra_sesion['Numero'] == num].shape[0]
                    fila[f'Frecuencia en {key}'] = freq_otra
                freq_general = df_historial_completo[df_historial_completo['Numero'] == num].shape[0]
                fila['Frecuencia General'] = freq_general
                temp_general = df_clasificacion_general[df_clasificacion_general['Numero'] == num]['Temperatura'].iloc[0]
                fila['Temperatura General'] = temp_general
                
                datos_comparacion.append(fila)
                
            df_comparacion = pd.DataFrame(datos_comparacion)
            st.dataframe(df_comparacion, width='stretch', hide_index=True)
            
            st.markdown("""
            **Cómo interpretar esta tabla:**
            - Compara si los números más fuertes de la sesión actual también lo son en el contexto general o en otras sesiones.
            - Un número 'Caliente' en una sesión pero 'Frío' en el general podría indicar un patrón específico de esa sesión.
            - Un número 'Caliente' en todas las categorías es un candidato muy robusto.
            """)

if __name__ == "__main__":
    main()