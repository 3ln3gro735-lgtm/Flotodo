# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import os
import traceback
import time 
from collections import defaultdict, Counter
import unicodedata

# --- CONFIGURACIÓN DE LA RUTA_RELATIVA ---
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
            st.warning("💡 **Solución:** Asegúrate de que el archivo 'Flotodo.csv' exista en la carpeta correcta.")
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
                if len(values) >= 5:
                    data.append(values)
        
        df_historial = pd.DataFrame(data, columns=column_names)
        
        if debug_mode:
            st.subheader("🔍 Examen de los Encabezados del CSV")
            st.write("**Lista completa de encabezados:**")
            st.code(header_line)
            st.dataframe(df_historial.head())
        
        df_historial.rename(columns={
            'Fecha': 'Fecha',
            'Tarde/Noche': 'Tipo_Sorteo',
            'Fijo': 'Fijo',
            '1er Corrido': 'Primer_Corrido',
            '2do Corrido': 'Segundo_Corrido'
        }, inplace=True)
        
        df_historial['Fecha'] = pd.to_datetime(df_historial['Fecha'], dayfirst=True, errors='coerce')
        df_historial.dropna(subset=['Fecha'], inplace=True)
        
        st.write("Normalizando la columna 'Tipo_Sorteo' para Florida (T y N)...")
        df_historial['Tipo_Sorteo'] = df_historial['Tipo_Sorteo'].astype(str).str.strip().str.upper().map({
            'TARDE': 'T', 'T': 'T',
            'NOCHE': 'N', 'N': 'N'
        }).fillna('OTRO')
        
        df_historial = df_historial[df_historial['Tipo_Sorteo'].isin(['T', 'N'])]
        
        st.success("Columna 'Tipo_Sorteo' normalizada (T, N).")
        if debug_mode:
            st.write("Valores únicos en 'Tipo_Sorteo' después de normalizar:", df_historial['Tipo_Sorteo'].unique())
        
        df_procesado = []
        for _, row in df_historial.iterrows():
            fecha = row['Fecha']
            tipo_sorteo = row['Tipo_Sorteo']
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

# --- FUNCIÓN PARA OBTENER ESTADO COMPLETO DE NÚMEROS (CORREGIDO: USA SOLO FIJO) ---
def get_full_state_dataframe(df_historial, fecha_referencia):
    st.info(f"📅 **Análisis de Estado:** Calculando el estado de todos los números (Solo Fijos) hasta la fecha **{fecha_referencia.strftime('%d/%m/%Y')}**.")
    
    # FILTRO IMPORTANTE: Solo usamos Fijos para calcular estados de números
    df_fijos_hist = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    df_fijos_filtrado = df_fijos_hist[df_fijos_hist['Fecha'] < fecha_referencia].copy()
    
    if df_fijos_filtrado.empty:
        return pd.DataFrame(), {}

    df_maestro = pd.DataFrame({'Numero': range(100)})
    primera_fecha_historica = df_fijos_hist['Fecha'].min()
    
    historicos_numero = {}
    for i in range(100):
        fechas_i = df_fijos_filtrado[df_fijos_filtrado['Numero'] == i]['Fecha'].sort_values()
        gaps = fechas_i.diff().dt.days.dropna()
        if len(gaps) > 0:
            historicos_numero[i] = gaps.median()
        else:
            historicos_numero[i] = (fecha_referencia - primera_fecha_historica).days

    df_maestro['Decena'] = df_maestro['Numero'] // 10
    df_maestro['Unidad'] = df_maestro['Numero'] % 10
    
    ultima_aparicion_num_key = df_fijos_filtrado.groupby('Numero')['Fecha'].max().reindex(range(100))
    ultima_aparicion_num_key.fillna(primera_fecha_historica, inplace=True)
    gap_num = (fecha_referencia - ultima_aparicion_num_key).dt.days
    df_maestro['Salto_Numero'] = gap_num
    df_maestro['Estado_Numero'] = df_maestro.apply(lambda row: calcular_estado_actual(row['Salto_Numero'], historicos_numero[row['Numero']]), axis=1)
    df_maestro['Última Aparición (Fecha)'] = ultima_aparicion_num_key.dt.strftime('%d/%m/%Y')
    
    # Frecuencia también basada solo en Fijos para consistencia
    frecuencia = df_fijos_filtrado['Numero'].value_counts().reindex(range(100)).fillna(0)
    df_maestro['Total_Salidas_Historico'] = frecuencia

    return df_maestro, historicos_numero

# --- FUNCIÓN PARA CLASIFICAR NÚMEROS POR TEMPERATURA ---
def crear_mapa_de_calor_numeros(df_frecuencia, top_n=30, medio_n=30):
    df_ordenado = df_frecuencia.sort_values(by='Total_Salidas_Historico', ascending=False).reset_index(drop=True).copy()
    df_ordenado['Temperatura'] = '🧊 Frío'
    if len(df_ordenado) > top_n:
        df_ordenado.loc[top_n : top_n + medio_n - 1, 'Temperatura'] = '🟡 Tibio'
        df_ordenado.loc[0 : top_n - 1, 'Temperatura'] = '🔥 Caliente'
    return df_ordenado

# --- FUNCIÓN PARA ANÁLISIS DE OPORTUNIDAD POR DÍGITO (CORREGIDO: USA SOLO FIJO) ---
def analizar_oportunidad_por_digito(df_historial, df_estados_completos, historicos_numero, modo_temperatura, fecha_inicio_rango, fecha_fin_rango, top_n_candidatos=5, fecha_referencia=None):
    st.info(f"🎯 **Análisis de Oportunidad por Dígito:** Iniciando análisis en modo: **{modo_temperatura}**.")
    
    # 1. Cálculo de TEMPERATURA (Solo Fijos para consistencia estadística)
    # Se filtra solo Fijo para evitar que los corridos inflen la temperatura
    df_base_fijos = df_historial[df_historial['Posicion'] == 'Fijo'].copy()

    if modo_temperatura == "Histórico Completo":
        df_temperatura = df_base_fijos.copy()
    else:
        end_of_day = fecha_fin_rango + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df_temperatura = df_base_fijos[(df_base_fijos['Fecha'] >= fecha_inicio_rango) & (df_base_fijos['Fecha'] <= end_of_day)].copy()
        if df_temperatura.empty:
            st.warning("El rango seleccionado no contiene sorteos. Se usará el historial completo.")
            df_temperatura = df_base_fijos.copy()

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
    if len(df_frecuencia_decenas) >= 7: df_frecuencia_decenas.loc[6:9, 'Temperatura'] = '🧊 Frío'
    
    df_frecuencia_unidades = df_frecuencia_unidades.sort_values(by='Frecuencia Total', ascending=False).reset_index(drop=True)
    df_frecuencia_unidades['Temperatura'] = '🟡 Tibio'
    if len(df_frecuencia_unidades) >= 3: df_frecuencia_unidades.loc[0:2, 'Temperatura'] = '🔥 Caliente'
    if len(df_frecuencia_unidades) >= 7: df_frecuencia_unidades.loc[6:9, 'Temperatura'] = '🧊 Frío'
    
    mapa_temp_decenas = pd.Series(df_frecuencia_decenas.Temperatura.values, index=df_frecuencia_decenas.Dígito).to_dict()
    mapa_temp_unidades = pd.Series(df_frecuencia_unidades.Temperatura.values, index=df_frecuencia_unidades.Dígito).to_dict()
    
    # 2. Cálculo de ESTADO (Normal/Vencido) - CORREGIDO: Usa solo Fijos
    df_hist_estado = df_base_fijos[df_base_fijos['Fecha'] < fecha_referencia].copy()
    
    def calcular_estado_digito_posicion(df, digito, tipo):
        if tipo == 'decena':
            fechas = df[df['Numero'] // 10 == digito]['Fecha'].sort_values()
        else: # unidad
            fechas = df[df['Numero'] % 10 == digito]['Fecha'].sort_values()
        
        if fechas.empty: return 'Normal', 0, 0
        
        gaps = fechas.diff().dt.days.dropna()
        promedio = gaps.median() if len(gaps) > 0 else 0
        ultima_fecha = fechas.max()
        gap_actual = (fecha_referencia - ultima_fecha).days
        estado = calcular_estado_actual(gap_actual, promedio)
        return estado, gap_actual, promedio

    resultados_decenas = []
    resultados_unidades = []
    
    for i in range(10):
        # Decenas
        estado_dec, gap_dec, prom_dec = calcular_estado_digito_posicion(df_hist_estado, i, 'decena')
        # Unidades
        estado_uni, gap_uni, prom_uni = calcular_estado_digito_posicion(df_hist_estado, i, 'unidad')

        # Puntuación
        puntuacion_base_decena = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}[estado_dec]
        puntuacion_base_unidad = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}[estado_uni]

        # Proactiva
        puntuacion_proactiva_decena = min(49, (gap_dec / prom_dec * 50) if prom_dec > 0 and estado_dec == 'Normal' else 0)
        puntuacion_proactiva_unidad = min(49, (gap_uni / prom_uni * 50) if prom_uni > 0 and estado_uni == 'Normal' else 0)

        # Temperatura
        puntuacion_temperatura_map = {'🔥 Caliente': 30, '🟡 Tibio': 20, '🧊 Frío': 10}
        temperatura_decena = mapa_temp_decenas.get(i, '🟡 Tibio')
        temperatura_unidad = mapa_temp_unidades.get(i, '🟡 Tibio')
        puntuacion_temp_decena = puntuacion_temperatura_map.get(temperatura_decena, 20)
        puntuacion_temp_unidad = puntuacion_temperatura_map.get(temperatura_unidad, 20)

        puntuacion_total_decena = puntuacion_base_decena + puntuacion_proactiva_decena + puntuacion_temp_decena
        puntuacion_total_unidad = puntuacion_base_unidad + puntuacion_proactiva_unidad + puntuacion_temp_unidad

        resultados_decenas.append({'Dígito': i, 'Rol': 'Decena', 'Temperatura': temperatura_decena, 'Estado': estado_dec, 'Puntuación Base': puntuacion_base_decena, 'Puntuación Proactiva': round(puntuacion_proactiva_decena, 1), 'Puntuación Temperatura': puntuacion_temp_decena, 'Puntuación Total': round(puntuacion_total_decena, 1)})
        resultados_unidades.append({'Dígito': i, 'Rol': 'Unidad', 'Temperatura': temperatura_unidad, 'Estado': estado_uni, 'Puntuación Base': puntuacion_base_unidad, 'Puntuación Proactiva': round(puntuacion_proactiva_unidad, 1), 'Puntuación Temperatura': puntuacion_temp_unidad, 'Puntuación Total': round(puntuacion_total_unidad, 1)})

    df_oportunidad_decenas = pd.DataFrame(resultados_decenas)
    df_oportunidad_unidades = pd.DataFrame(resultados_unidades)

    # Candidatos combinados
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

    return df_oportunidad_decenas, df_oportunidad_unidades, df_candidatos, mapa_temp_decenas, mapa_temp_unidades

# --- FUNCIÓN PARA AUDITORÍA HISTÓRICA (CORREGIDO: USA SOLO FIJO) ---
def generar_auditoria_doble_normal(df_historial):
    st.info("🔍 **Auditoría Histórica:** Buscando los últimos 100 eventos 'Doble Normal' (Solo Fijos).")
    
    df_fijo = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    if df_fijo.empty:
        st.warning("No hay datos para auditoría.")
        return pd.DataFrame()

    fechas_unicas = df_fijo['Fecha'].unique()
    fechas_unicas = np.sort(fechas_unicas)
    
    if len(fechas_unicas) > 100:
        fechas_unicas = fechas_unicas[-100:]
        st.caption("ℹ️ Nota: El análisis de auditoría se limita a los últimos 100 eventos.")
    
    auditoria = []
    
    for i, fecha in enumerate(fechas_unicas):
        df_hasta_ahora = df_fijo[df_fijo['Fecha'] < fecha].copy()
        sorteos_fecha = df_fijo[df_fijo['Fecha'] == fecha].sort_values(by='Tipo_Sorteo') 
        
        for _, row in sorteos_fecha.iterrows():
            numero = row['Numero']
            decena = numero // 10
            unidad = numero % 10
            
            # Estado Decena
            fechas_dec = df_hasta_ahora[df_hasta_ahora['Numero'] // 10 == decena]['Fecha'].sort_values()
            gaps_dec = fechas_dec.diff().dt.days.dropna()
            prom_dec = gaps_dec.median() if len(gaps_dec) > 0 else 0
            gap_dec_actual = (fecha - fechas_dec.max()).days if not fechas_dec.empty else 0
            estado_dec = calcular_estado_actual(gap_dec_actual, prom_dec)
            
            # Estado Unidad
            fechas_uni = df_hasta_ahora[df_hasta_ahora['Numero'] % 10 == unidad]['Fecha'].sort_values()
            gaps_uni = fechas_uni.diff().dt.days.dropna()
            prom_uni = gaps_uni.median() if len(gaps_uni) > 0 else 0
            gap_uni_actual = (fecha - fechas_uni.max()).days if not fechas_uni.empty else 0
            estado_uni = calcular_estado_actual(gap_uni_actual, prom_uni)
            
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

# --- FUNCIÓN PARA BUSCAR PATRONES (CORREGIDO: USA SOLO FIJO) ---
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

# --- FUNCIÓN TABLA HISTÓRICO VISUAL (CORREGIDO: USA SOLO FIJO) ---
def crear_tabla_historico_visual_fijo(df_historial, num_ultimos=30):
    df_fijo = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    if df_fijo.empty:
        return pd.DataFrame()

    df_fijo['Fecha'] = pd.to_datetime(df_fijo['Fecha'])
    
    fechas_unicas = df_fijo['Fecha'].unique()
    fechas_ordenadas = sorted(fechas_unicas, reverse=True) 
    ultimas_fechas = fechas_ordenadas[:num_ultimos]
    
    historial_visual = []
    
    for fecha in ultimas_fechas:
        for sesion_key, sesion_nombre in [('N', 'Noche'), ('T', 'Tarde')]:
            resultado = df_fijo[(df_fijo['Fecha'] == fecha) & (df_fijo['Tipo_Sorteo'] == sesion_key)]
            if not resultado.empty:
                numero = resultado.iloc[0]['Numero']
                
                df_hasta_fecha = df_fijo[df_fijo['Fecha'] < fecha].copy()
                
                # Calculamos estado Decena
                fechas_dec = df_hasta_fecha[df_hasta_fecha['Numero'] // 10 == (numero // 10)]['Fecha'].sort_values()
                gaps_dec = fechas_dec.diff().dt.days.dropna()
                prom_dec = gaps_dec.median() if len(gaps_dec) > 0 else 0
                gap_dec = (fecha - fechas_dec.max()).days if not fechas_dec.empty else 0
                estado_dec = calcular_estado_actual(gap_dec, prom_dec)
                
                # Calculamos estado Unidad
                fechas_uni = df_hasta_fecha[df_hasta_fecha['Numero'] % 10 == (numero % 10)]['Fecha'].sort_values()
                gaps_uni = fechas_uni.diff().dt.days.dropna()
                prom_uni = gaps_uni.median() if len(gaps_uni) > 0 else 0
                gap_uni = (fecha - fechas_uni.max()).days if not fechas_uni.empty else 0
                estado_uni = calcular_estado_actual(gap_uni, prom_uni)
                
                es_doble_normal = (estado_dec == 'Normal' and estado_uni == 'Normal')
                
                # Frecuencia
                df_freq_total = df_hasta_fecha['Numero'].value_counts().reset_index()
                df_freq_total.columns = ['Numero', 'Total_Salidas_Historico']
                df_freq = crear_mapa_de_calor_numeros(df_freq_total)
                row = df_freq[df_freq['Numero'] == numero]
                temp = row['Temperatura'].iloc[0] if not row.empty else 'N/A'

                # Estado número completo
                fechas_num = df_hasta_fecha[df_hasta_fecha['Numero'] == numero]['Fecha'].sort_values()
                gaps_num = fechas_num.diff().dt.days.dropna()
                prom_num = gaps_num.median() if len(gaps_num) > 0 else 0
                gap_num_val = (fecha - fechas_num.max()).days if not fechas_num.empty else 0
                estado_num = calcular_estado_actual(gap_num_val, prom_num)

                historial_visual.append({
                    'Fecha': fecha.strftime('%d/%m/%Y'),
                    'Sesión': sesion_nombre,
                    'Fijo': f"{numero:02d}",
                    'Es Doble Normal': es_doble_normal,
                    'Temperatura': temp,
                    'Estado del Número': estado_num
                })

    return pd.DataFrame(historial_visual)

# --- FUNCIÓN ESTRATEGIA TENDENCIA (Ya usaba solo Fijo, se mantiene) ---
def generar_estrategia_tendencia(df_historial, fecha_referencia):
    df_fijo = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    df_fijo_analisis = df_fijo[df_fijo['Fecha'] < fecha_referencia].copy()
    
    if df_fijo_analisis.empty: return pd.DataFrame(), [], [], []

    # Estados dígitos (Posicionales)
    estados_decena = {}
    estados_unidad = {}
    
    for d in range(10):
        # Decena
        fechas_dec = df_fijo_analisis[df_fijo_analisis['Numero'] // 10 == d]['Fecha'].sort_values()
        if len(fechas_dec) > 0:
            gaps = fechas_dec.diff().dt.days.dropna()
            prom = gaps.median() if len(gaps) > 0 else 0
            gap_act = (fecha_referencia - fechas_dec.max()).days
            estados_decena[d] = {'estado': calcular_estado_actual(gap_act, prom), 'gap': gap_act, 'promedio': prom}
        else:
            estados_decena[d] = {'estado': 'Normal', 'gap': 0, 'promedio': 0}

        # Unidad
        fechas_uni = df_fijo_analisis[df_fijo_analisis['Numero'] % 10 == d]['Fecha'].sort_values()
        if len(fechas_uni) > 0:
            gaps = fechas_uni.diff().dt.days.dropna()
            prom = gaps.median() if len(gaps) > 0 else 0
            gap_act = (fecha_referencia - fechas_uni.max()).days
            estados_unidad[d] = {'estado': calcular_estado_actual(gap_act, prom), 'gap': gap_act, 'promedio': prom}
        else:
            estados_unidad[d] = {'estado': 'Normal', 'gap': 0, 'promedio': 0}

    # Estados números
    estados_numero = {n: {'estado': 'Normal', 'gap': 0, 'promedio': 0} for n in range(100)}
    for n in range(100):
        fechas_num = df_fijo_analisis[df_fijo_analisis['Numero'] == n]['Fecha'].sort_values()
        gaps = fechas_num.diff().dt.days.dropna()
        if len(gaps) > 0:
            promedio_gap = gaps.median()
            ultima_fecha_num = fechas_num.max()
            gap_actual = (fecha_referencia - ultima_fecha_num).days
            estados_numero[n] = {'estado': calcular_estado_actual(gap_actual, promedio_gap), 'gap': gap_actual, 'promedio': promedio_gap}

    datos_completa = []
    for num in range(100):
        decena, unidad = num // 10, num % 10
        if (estados_decena[decena]['estado'] == 'Normal' and 
            estados_unidad[unidad]['estado'] == 'Normal'):
            
            estado_num = estados_numero[num]['estado']
            datos_completa.append({
                'Número': num, 
                'Estado del Número': estado_num,
                'Salto (Días)': estados_numero[num]['gap']
            })
            
    df_completa = pd.DataFrame(datos_completa)

    candidatos_vencidos = []
    if not df_completa.empty:
        vencidos_mask = df_completa['Estado del Número'].isin(['Vencido', 'Muy Vencido'])
        candidatos_vencidos = df_completa[vencidos_mask]['Número'].tolist()

    candidatos_normales = []
    if not df_completa.empty:
        normales_mask = df_completa['Estado del Número'] == 'Normal'
        df_potenciales = df_completa[normales_mask]['Número'].tolist()
        
        for num in df_potenciales:
            fechas_num = df_fijo_analisis[df_fijo_analisis['Numero'] == num]['Fecha'].sort_values().tolist()
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
                    candidatos_normales.append(num)
    
    candidatos_normales = sorted(candidatos_normales)[:10]
    candidatos_finales = sorted(list(set(candidatos_vencidos + candidatos_normales)))
    
    return df_completa, candidatos_finales, sorted(candidatos_vencidos), sorted(candidatos_normales)

# --- FUNCIÓN PRINCIPAL ---
def main():
    st.sidebar.header("⚙️ Opciones de Análisis - Florida")
    
    with st.sidebar.expander("📝 Agregar Nuevo Sorteo (Actualizar CSV)", expanded=False):
        st.caption("Actualiza los resultados rápidamente.")
        
        fecha_nueva = st.date_input("Fecha del sorteo:", value=datetime.now().date(), format="DD/MM/YYYY")
        
        sesion = st.radio("Sesión:", ["Tarde (T)", "Noche (N)"], horizontal=True)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            fijo = st.number_input("Fijo", min_value=0, max_value=99, value=0, format="%02d")
        with col_b:
            p1 = st.number_input("1er Corr.", min_value=0, max_value=99, value=0, format="%02d")
        with col_c:
            p2 = st.number_input("2do Corr.", min_value=0, max_value=99, value=0, format="%02d")
        
        if st.button("💾 Guardar Sorteo", type="primary"):
            sesion_code = "T" if "Tarde" in sesion else "N"
            fecha_str = fecha_nueva.strftime('%d/%m/%Y')
            linea_nueva = f"{fecha_str};{sesion_code};{fijo};{p1};{p2}\n"
            
            try:
                carpeta_csv = os.path.dirname(RUTA_CSV)
                if carpeta_csv and not os.path.exists(carpeta_csv):
                    os.makedirs(carpeta_csv)
                
                with open(RUTA_CSV, 'a', encoding='latin-1') as f:
                    f.write(linea_nueva)
                
                st.success("✅ ¡Sorteo guardado!")
                st.info("Actualizando gráficos...")
                
                st.cache_resource.clear()
                time.sleep(1.5)
                st.rerun()
                
            except PermissionError:
                st.error("❌ Error de permisos: Asegúrate de que el archivo CSV no esté abierto en Excel.")
            except Exception as e:
                st.error(f"❌ Error al guardar: {str(e)}")

    debug_mode = st.sidebar.checkbox("🔍 Activar Modo Diagnóstico (CSV)", value=False)
    
    st.sidebar.subheader("📊 Modo de Análisis de Datos")
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
    st.sidebar.subheader("🌡️ Modo de temperatura de Dígitos (Almanaque)")
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
        # Mostrar info de último sorteo (usando Fijo)
        df_temp_fijo = df_analisis[df_analisis['Posicion'] == 'Fijo']
        if 'T' in df_temp_fijo['Tipo_Sorteo'].unique():
            ultimo_sorteo_T = df_temp_fijo[df_temp_fijo['Tipo_Sorteo'] == 'T'].iloc[-1]
            st.sidebar.info(f"Último sorteo **Tarde**: {ultimo_sorteo_T['Fecha'].strftime('%d/%m/%Y')} (Fijo: {ultimo_sorteo_T['Numero']})")
        if 'N' in df_temp_fijo['Tipo_Sorteo'].unique():
            ultimo_sorteo_N = df_temp_fijo[df_temp_fijo['Tipo_Sorteo'] == 'N'].iloc[-1]
            st.sidebar.info(f"Último sorteo **Noche**: {ultimo_sorteo_N['Fecha'].strftime('%d/%m/%Y')} (Fijo: {ultimo_sorteo_N['Numero']})")
        
        df_estados_completos, historicos_numero = get_full_state_dataframe(df_analisis, fecha_referencia)
        
        if df_estados_completos.empty:
            st.error("No se pudo calcular el estado de los números para la fecha de referencia.")
            st.stop()

        frecuencia_numeros_historica = df_estados_completos[['Numero', 'Total_Salidas_Historico']].copy() # Ya viene filtrado por Fijo
        df_clasificacion_actual = crear_mapa_de_calor_numeros(frecuencia_numeros_historica)
        
        fecha_inicio_rango_safe = pd.to_datetime(fecha_inicio_rango).tz_localize(None) if fecha_inicio_rango else None
        fecha_fin_rango_safe = pd.to_datetime(fecha_fin_rango).tz_localize(None) if fecha_fin_rango else None
        
        df_oportunidad_decenas, df_oportunidad_unidades, top_candidatos, mapa_temp_decenas, mapa_temp_unidades = analizar_oportunidad_por_digito(
            df_analisis, df_estados_completos, historicos_numero, 
            modo_temperatura, fecha_inicio_rango_safe, fecha_fin_rango_safe,
            top_n_candidatos, fecha_referencia=fecha_referencia
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
        st.header(f"🌡️ Clasificación de Números (Basada en {modo_sorteo} - Solo Fijos)")
        
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

        # --- SECCIÓN 3: ANÁLISIS DE OPORTUNIDAD POR DÍGITO ---
        st.markdown("---")
        st.header("🎯 Análisis de Oportunidad por Dígito (Decenas y Unidades)")
        st.info("Esta tabla ahora coincide exactamente con la lógica de 'Doble Normal' usada en la Estrategia de Tendencia (Basada solo en Fijos).")
        
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
        
        # --- SECCIÓN 4: AUDITORÍA HISTÓRICA ---
        st.markdown("---")
        st.header("🔍 Auditoría Histórica de 'Doble Normal' (Corroboración de Aciertos)")
        st.markdown("Historia de los últimos 100 eventos 'Doble Normal' y el tiempo transcurrido desde el anterior.")
        
        df_auditoria = generar_auditoria_doble_normal(df_analisis)
        
        if not df_auditoria.empty:
            st.dataframe(df_auditoria, width='stretch', hide_index=True)
        
        # --- SECCIÓN 5: PATRONES SECUENCIALES ---
        st.markdown("---")
        st.header("🔍 Búsqueda de Patrones Secuenciales")
        st.markdown("Busca patrones en la columna Fijo.")
        df_analisis_para_patrones = df_analisis[df_analisis['Fecha'] < fecha_referencia].copy()
        nombre_sesion_para_patrones = modo_sorteo.split(':')[-1].strip()
        patrones = buscar_patrones_secuenciales(df_analisis_para_patrones, max_longitud=3, nombre_sesion=nombre_sesion_para_patrones)
        
        if patrones:
            st.subheader("Últimos Patrones Detectados y Posibles Siguientes")
            df_fijo_patron = df_analisis_para_patrones[df_analisis_para_patrones['Posicion'] == 'Fijo'].copy()
            ultimos_numeros = df_fijo_patron.tail(3)['Numero'].tolist()
            
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

        # --- SECCIÓN 6: MATRIZ COMPLETA Y CANDIDATOS ---
        st.markdown("---")
        st.header("🧠 Generador y Evaluador de Estrategia de Tendencia")
        
        st.subheader(f"Candidatos para el {fecha_referencia.strftime('%d/%m/%Y')}")
        
        df_completa, candidatos_finales, candidatos_vencidos, candidatos_normales = generar_estrategia_tendencia(df_analisis, fecha_referencia)
        
        if not df_completa.empty:
            # Verificar aciertos solo con Fijos
            df_fijos_hoy = df_analisis[(df_analisis['Fecha'] == fecha_referencia) & (df_analisis['Posicion'] == 'Fijo')]
            resultados_del_dia = df_fijos_hoy['Numero'].tolist()
            
            df_completa['¿Salió Hoy?'] = df_completa['Número'].isin(resultados_del_dia)
            
            def resaltar_aciertos(row):
                if row['¿Salió Hoy?']:
                    return ['background-color: #00FF00; color: black; font-weight: bold' for _ in row]
                else:
                    return ['' for _ in row]
            
            df_completa['Número'] = df_completa['Número'].apply(lambda x: f"{x:02d}")
            
            st.markdown("### 🔵 Listado Completo Normal-Normal (Con Estados)")
            st.info("📋 Esta lista muestra todos los números con Decena/Unidad Normales. Los marcados en VERDE salieron hoy.")
            
            st.dataframe(
                df_completa.style.apply(resaltar_aciertos, axis=1), 
                use_container_width=True,
                hide_index=True
            )
            
            aciertos_lista = df_completa['¿Salió Hoy?'].sum()
            if aciertos_lista > 0:
                st.success(f"🎯 ¡Se acertaron {aciertos_lista} números de esta lista en los sorteos de hoy!")
            else:
                st.write("ℹ️ Ningún número de esta lista salió hoy.")
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

        # --- SECCIÓN 7: HISTORIAL VISUAL ENRIQUECIDA ---
        st.markdown("---")
        st.header("📊 Historial Visual Enriquecido (Últimos 30 Fijos)")
        st.markdown("Tabla ordenada cronológicamente. El evento más reciente (Noche de hoy) aparece arriba.")
        
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

if __name__ == "__main__":
    main()