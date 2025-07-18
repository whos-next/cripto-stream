# Predição de Criptomoedas com Streamlit

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-orange.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Ready-brightgreen.svg)](https://scikit-learn.org/)

Uma aplicação web interativa para visualizar, analisar e prever os preços de criptomoedas usando múltiplos modelos de Machine Learning.

## Sobre o Projeto

[![Aplicação](https://i.postimg.cc/jd3QKZSq/Crip2.png)](https://postimg.cc/vc96t7nC)

Este projeto foi desenvolvido para fornecer uma ferramenta intuitiva que permite a qualquer pessoa, mesmo sem conhecimento profundo em programação ou finanças, treinar modelos de previsão para o mercado de criptoativos. A interface permite customizar os dados de entrada, os parâmetros dos modelos e comparar visualmente os resultados.

### Recursos Principais

- **Análise Interativa:** Selecione diferentes criptomoedas e períodos de tempo.
- **Features Enriquecidas:** Adicione ativos financeiros correlacionados (como S&P 500, DXY, etc.) para melhorar a precisão do modelo.
- **Engenharia de Features Customizável:** Defina janelas de tempo para *lags* e médias móveis.
- **Múltiplos Modelos:** Escolha entre Regressão Linear, Random Forest, Redes Neurais (MLP), XGBoost e SVM.
- **Comparação de Modelos:** Treine dois algoritmos diferentes com os mesmos dados e compare suas performances lado a lado.
- **Exportação de Dados:** Baixe os dados históricos e as previsões em formato CSV.

### Construído Com

*   [Streamlit](https://streamlit.io/)
*   [Pandas](https://pandas.pydata.org/)
*   [Scikit-learn](https://scikit-learn.org/)
*   [yfinance](https://pypi.org/project/yfinance/)

## Começando

### Acesso Online
A aplicação está disponível online e pode ser acessada diretamente em:

**🌐 [https://criptostream.streamlit.app/](https://criptostream.streamlit.app/)**

Ou siga os passos abaixo para executar a aplicação localmente.

### Pré-requisitos

- Python 3.9 ou superior

### Instalação

1.  Clone o repositório:
    ```bash
    git clone https://github.com/whos-next/cripto-stream.git
    ```
2.  Navegue até o diretório do projeto e instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## Uso local


Para iniciar a aplicação, execute o seguinte comando no seu terminal:

```bash
streamlit run main.py
```

A interface será aberta no seu navegador (http://localhost:8501/). Para usar:

1.  **Defina o Período:** Selecione as datas de início e fim para os dados históricos.
2.  **Escolha os Ativos:** Selecione a criptomoeda principal e, opcionalmente, ativos auxiliares.
3.  **Configure as Features:** Ajuste os parâmetros de *lags* e médias móveis.
4.  **Selecione um Modelo:** Escolha um dos algoritmos de Machine Learning.
5.  **Treine e Analise:** Clique em "Treinar Modelo" para ver os gráficos, métricas e previsões.

### Snippets de Código

As funções de obtenção de dados podem ser importadas e usadas em outros scripts Python:

```python
from main import obter_dados_cripto, obter_dados_multiplos
import datetime

# Definir período
inicio = datetime.date(2021, 1, 1)
final = datetime.date(2021, 12, 31)

# Obter dados apenas do Bitcoin
df_btc = obter_dados_cripto("Bitcoin", inicio, final)
print("Dados do Bitcoin:")
print(df_btc.head())

# Obter dados combinados
df_multi = obter_dados_multiplos(
    "Ethereum",
    ["Bitcoin", "S&P 500"],
    inicio,
    final
)
print("\nDados combinados:")
print(df_multi.head())
```


## Contribuição

Contribuições são o que tornam a comunidade de código aberto um lugar incrível para aprender, inspirar e criar. Qualquer contribuição que você fizer será **muito apreciada**.
