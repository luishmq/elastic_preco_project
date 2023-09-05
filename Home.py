import streamlit as st

st.set_page_config(
    page_title="Previsão de Elasticidade de Preço",
    page_icon="📈",
    layout='wide'
)

st.title("Bem-vindo(a) ao painel do Projeto de Previsão de Elasticidade de Preço! 📈")

st.markdown(
    """
    Esta aplicação foi desenvolvida para um e-commerce de produtos eletrônicos que solicitou um modelo de **Inteligência Artificial** capaz de prever a elasticidade de preço. 
    
    - A elasticidade de preço é um conceito econômico que mede a sensibilidade da demanda de um produto às mudanças de preço. 
    - Com base nessa previsão, você pode tomar decisões estratégicas e maximizar o faturamento do seu negócio.

    ##### Com esse modelo, você pode simular cenários de aumento de preço e descontos e receber feedbacks em tempo real sobre os impactos financeiros no faturamento.


    Tenha acesso a insights valiosos para otimizar a estratégia de preços do seu negócio. Ao compreender a elasticidade de preço dos seus produtos, você poderá ajustar os preços de forma inteligente, aproveitando oportunidades de aumentar as vendas e **maximizar o faturamento**.

    **👈 Selecione uma demonstração na barra lateral** para ver exemplos e experimentar você mesmo!
    """
)