# Tratamento de dados
from scipy import stats as st
import numpy as np
import pandas as pd

eventos = pd.read_csv('/datasets/ab_project_marketing_events_us.csv')
novos_usuarios = pd.read_csv('/datasets/final_ab_new_users_upd_us.csv')
eventos_novos_usuarios = pd.read_csv('/datasets/final_ab_events_upd_us.csv')
participantes_teste = pd.read_csv('/datasets/final_ab_participants_upd_us.csv')

#Alterando tipo

eventos['start_dt'] = pd.to_datetime(eventos['start_dt'])
eventos['finish_dt'] = pd.to_datetime(eventos['finish_dt'])
novos_usuarios['first_date'] = pd.to_datetime(novos_usuarios['first_date'])
eventos_novos_usuarios['event_dt'] = pd.to_datetime(eventos_novos_usuarios['event_dt'])

# Análise Exploratória de Dados
# Filtrando premissas
usuarios_eu = novos_usuarios.query("region == 'EU'")['user_id']
usuarios_filtrados = eventos_novos_usuarios[eventos_novos_usuarios['user_id'].isin(usuarios_eu)]
data_experimento = usuarios_filtrados.query("event_dt >= '2020-12-07' and event_dt <= '2021-01-01'")
first_date = data_experimento.merge(novos_usuarios, on= 'user_id', how='right')
session_date = first_date[first_date['event_dt'] - first_date['first_date'] <= pd.Timedelta('14 days')]

# Encontrando nº de usuários únicos em diferentes etapas do funil
usuarios_product_page = session_date[session_date['event_name'] == 'product_page']['user_id'].nunique()
usuarios_product_cart = session_date[session_date['event_name']== 'product_cart']['user_id'].nunique()
usuarios_purchase = session_date[session_date['event_name'] == 'purchase']['user_id'].nunique()
print('Usuarios Product Page:', usuarios_product_page)
print('Usuarios Product Cart:', usuarios_product_cart)
print('Usuarios Purchase:', usuarios_purchase)
print()

sem_carrinho = usuarios_purchase - usuarios_product_cart
print('Usuários que compraram sem carrinho:', sem_carrinho)

ids_carrinho = set(session_date[session_date['event_name'] == 'product_cart']['user_id'])
ids_compra = set(session_date[session_date['event_name'] == 'purchase']['user_id'])

#O número de eventos por usuário é distribuído igualmente entre as amostras?
df = session_date.merge(participantes_teste, on='user_id', how='inner')
eventos_por_usuario = df.groupby(['user_id', 'event_name','ab_test']).size().reset_index(name='num_eventos')
media_eventos_por_grupo = eventos_por_usuario.groupby('ab_test')['num_eventos'].mean().reset_index()
print()
print('A média de eventos por usuário:', media_eventos_por_grupo)

#Limpando e filtrando usuarios unicos p/ teste A/B
usuarios_por_grupo = df.groupby('user_id')['ab_test'].nunique()
multigrupo = usuarios_por_grupo[usuarios_por_grupo > 1].reset_index()
df_limpo = df[~df['user_id'].isin(multigrupo['user_id'])]

#Os usuarios de ambas as amostras estao presentes?
test = df_limpo[df_limpo['group'].isin(['A', 'B'])].copy()

# Como o numero de eventos é distribuido entre os dias
test['data_evento'] = test['event_dt'].dt.date
eventos_por_data = test.groupby('data_evento').size().reset_index(name='num_eventos')

# Excluindo data incompleta e filtrando 
data_filtro = pd.to_datetime('2020-12-30').date()
data_incompleta = test[test['data_evento'] == data_filtro]
df_limpo = test[test['data_evento'] != data_filtro]
test = df_limpo[df_limpo['ab_test'].isin(['recommender_system_test'])]

import plotly.express as px

session_date = pd.DataFrame({
    "stage": ["Product Page", "Product Cart", "Purchase"],
    "number": [usuarios_product_page, usuarios_product_cart, usuarios_purchase]
})

fig = px.funnel(session_date, x="number", y="stage")
fig.show()

# - Entre a etapa de conversão do carrinho para a compra efetiva, observa-se uma diferença de 877 transações. O indicador Product Cart registra 14.278 inclusões, enquanto o indicador Purchase contabiliza 15.155 compras concluídas — exatamente 877 a mais do que os itens adicionados ao carrinho.

- Os usuários de ambas as amostras estão presentes em uma nova variável chamada test.

- O número dos eventos estão distribuidos consideralmente parecidos (com bastante eventos), exceto dia 30/12/2020 que está com dados incompletos, portanto foi retirado p/ não interferir no teste.

# TESTE A/B
# #comparando taxa de conversão
usuarios_product_page1 = test[test['event_name'] == 'product_page']['user_id'].nunique()
usuarios_product_cart1 = test[test['event_name'] == 'product_cart']['user_id'].nunique()
usuarios_purchase1 = test[test['event_name'] == 'purchase']['user_id'].nunique()

# separando por grupos
testA = test[test['group'] == 'A']
testB = test[test['group'] == 'B']

usuarios_product_page_A = testA[testA['event_name'] == 'product_page']['user_id'].nunique()
usuarios_product_cart_A = testA[testA['event_name'] == 'product_cart']['user_id'].nunique()
usuarios_purchase_A     = testA[testA['event_name'] == 'purchase']['user_id'].nunique()

usuarios_product_page_B = testB[testB['event_name'] == 'product_page']['user_id'].nunique()
usuarios_product_cart_B = testB[testB['event_name'] == 'product_cart']['user_id'].nunique()
usuarios_purchase_B     = testB[testB['event_name'] == 'purchase']['user_id'].nunique()

taxa_cart_A = usuarios_product_cart_A / usuarios_product_page_A
taxa_purchase_A = usuarios_purchase_A / usuarios_product_page_A
print("Taxa de conversão A (Page → Cart):", taxa_cart_A)
print("Taxa de conversão A (Page → Purchase):", taxa_purchase_A)

print()

taxa_cart_B = usuarios_product_cart_B / usuarios_product_page_B
taxa_purchase_B = usuarios_purchase_B / usuarios_product_page_B
print("Taxa de conversão B (Page → Cart):", taxa_cart_B)
print("Taxa de conversão B (Page → Purchase):", taxa_purchase_B)

# Z-test
from statsmodels.stats.proportion import proportions_ztest

def ztest_funil(test, etapas):
    testA = test[test['group'] == 'A']
    testB = test[test['group'] == 'B']
    
    total_A = testA[testA['event_name'] == 'login']['user_id'].nunique()
    total_B = testB[testB['event_name'] == 'login']['user_id'].nunique()
    
    resultados = []
    
    for etapa in etapas:
        usuarios_A = testA[testA['event_name'] == etapa]['user_id'].nunique()
        usuarios_B = testB[testB['event_name'] == etapa]['user_id'].nunique()
        
        sucessos = [usuarios_A, usuarios_B]
        totais   = [total_A, total_B]
        z_stat, p_value = proportions_ztest(sucessos, totais)
        
        resultados.append({
            "etapa": etapa,
            "z_stat": z_stat,
            "p_value": p_value
        })
        
        print(f"Etapa: {etapa}")
        print("Z-statistic:", z_stat)
        print("p-value:", p_value)
        print()
    
    return resultados
etapas = ["product_page", "product_cart", "purchase"]
resultados = ztest_funil(test, etapas)

# Conclusões finais
Após deixar os dados consistentes para teste A/B, fiz a distribuição entre os grupos e o teste apresentou comportamento semelhante, indicando equilíbrio entre as amostras e reduzindo riscos de viés na comparação dos resultados.
Identifiquei que o dia 30/12/2020 possuía dados incompletos. A remoção dessa data foi uma decisão importante para evitar distorções estatísticas e garantir maior confiabilidade no teste.
Outro ponto observado foi a diferença entre os eventos de Carrinho e Compra. Enquanto o indicador de carrinho registrou 14.278 inclusões, o indicador de compra contabilizou 15.155 compras concluídas, ou seja, 877 compras a mais do que itens adicionados ao carrinho, o que pode sugerir que alguns usuários fazem compras sem a necessidade de passar pelo carrinho.
Em termos descritivos, o grupo B apresentou desempenho superior ao grupo A em ambas as etapas do funil, sugerindo inicialmente que a nova versão poderia gerar melhores resultados de conversão. Mas apesar de o grupo B apresentar taxas de conversão numericamente maiores, os testes estatísticos mostraram que essas diferenças não foram estatisticamente significativas então, como ambos os valores de p-value são maiores que 0,05, não há evidências estatísticas suficientes para comprovar diferença real entre os grupos.
