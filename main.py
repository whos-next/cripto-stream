import streamlit as st
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def params_rl(suffix=""):
    st.write("### Parâmetros da Regressão Linear")
    alpha = st.slider("Alpha", 0.0, 1.0, 0.01, 0.01, key=f"alpha{suffix}")
    fit_intercept = st.checkbox("Fit Intercept", value=True, key=f"fit_intercept{suffix}")
    return alpha, fit_intercept


def params_rf(suffix=""):
    """
    Exibe e captura os parâmetros para o modelo Random Forest.

    Args:
        suffix (str): Um sufixo opcional para as chaves dos widgets do Streamlit,
                      permitindo que múltiplos widgets do mesmo tipo coexistam.

    Returns:
        tuple: Uma tupla contendo o número de estimadores, a profundidade máxima
               e o número mínimo de amostras para dividir.
    """
    st.write("### Parâmetros do Random Forest")
    n_estimators = st.slider("Número de Estimadores", 10, 200, 100, 10, key=f"n_estimators_rf{suffix}")
    max_depth = st.slider("Profundidade Máxima", 1, 20, 10, 1, key=f"max_depth_rf{suffix}")
    min_samples_split = st.slider("Mínimo de Amostras para Dividir", 2, 20, 2, 1, key=f"min_samples_split{suffix}")
    return n_estimators, max_depth, min_samples_split


def params_mlp(suffix=""):
    """
    Exibe e captura os parâmetros para o modelo de Redes Neurais (MLP).

    Args:
        suffix (str): Um sufixo opcional para as chaves dos widgets do Streamlit.

    Returns:
        tuple: Uma tupla contendo os tamanhos das camadas ocultas, a função de ativação
               e o solver.
    """
    st.write("### Parâmetros das Redes Neurais (MLP)")
    hidden_layer_sizes = st.text_input("Quantidade Neuronios nas Camadas Ocultas", "100, 50", key=f"hidden_layers{suffix}")
    activation = st.selectbox("Função de Ativação", ["relu", "tanh", "logistic"], key=f"activation{suffix}")
    solver = st.selectbox("Solver", ["adam", "sgd", "lbfgs"], key=f"solver{suffix}")
    return hidden_layer_sizes, activation, solver


def params_gb(suffix=""):
    """
    Exibe e captura os parâmetros para o modelo XGBoost.

    Args:
        suffix (str): Um sufixo opcional para as chaves dos widgets do Streamlit.

    Returns:
        tuple: Uma tupla contendo o número de estimadores, a taxa de aprendizado
               e a profundidade máxima.
    """
    st.write("### Parâmetros do XGBoost")
    n_estimators = st.slider("Número de Estimadores", 10, 200, 100, 10, key=f"n_estimators_gb{suffix}")
    learning_rate = st.slider("Taxa de Aprendizado", 0.01, 0.5, 0.1, 0.01, key=f"learning_rate{suffix}")
    max_depth = st.slider("Profundidade Máxima", 1, 20, 10, 1, key=f"max_depth_gb{suffix}")
    return n_estimators, learning_rate, max_depth


def params_svm(suffix=""):
    """
    Exibe e captura os parâmetros para o modelo SVM.

    Args:
        suffix (str): Um sufixo opcional para as chaves dos widgets do Streamlit.

    Returns:
        tuple: Uma tupla contendo o parâmetro de regularização C, o kernel
               e o coeficiente de kernel gamma.
    """
    st.write("### Parâmetros do SVM")
    C = st.slider("C", 0.01, 10.0, 1.0, 0.01, key=f"C{suffix}")
    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], key=f"kernel{suffix}")
    gamma = st.selectbox("Gamma", ["scale", "auto"], key=f"gamma{suffix}")
    return C, kernel, gamma


def obter_dados_crypto(simbolo, data_inicio, data_final):
    """
    Busca os dados históricos de fechamento para um determinado símbolo de criptomoeda
    ou ativo financeiro usando a biblioteca yfinance.

    Args:
        simbolo (str): O nome do ativo (ex: "Bitcoin", "S&P 500").
        data_inicio (datetime.date): A data de início para a busca de dados.
        data_final (datetime.date): A data final para a busca de dados.

    Returns:
        pd.DataFrame: Um DataFrame com as colunas 'Data' e 'Fechamento'.
    """
    mapeamento_simbolos = {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Binance Coin": "BNB-USD",
        "Solana": "SOL-USD",
        "Cardano": "ADA-USD",
        # Ativos auxiliares adicionais
        "S&P 500": "^GSPC",
        "DXY": "DX-Y.NYB",
        "Nasdaq 100": "^NDX",
        "Tesla": "TSLA",
        "Apple": "AAPL",
    }
    ticker = mapeamento_simbolos.get(simbolo, "BTC-USD")
    dados = yf.download(ticker, start=data_inicio, end=data_final)
    dados.reset_index(inplace=True)
    dados = dados[["Date", "Close"]].rename(columns={"Date": "Data", "Close": "Fechamento"})
    return dados


def obter_dados_multiplos(cripto_principal, ativos_auxiliares, data_inicio, data_final):
    """
    Obtém os dados da criptomoeda principal e os combina com os dados de ativos auxiliares.

    Args:
        cripto_principal (str): A criptomoeda principal a ser analisada.
        ativos_auxiliares (list): Uma lista de nomes de ativos auxiliares.
        data_inicio (datetime.date): A data de início para a busca de dados.
        data_final (datetime.date): A data final para a busca de dados.

    Returns:
        pd.DataFrame: Um DataFrame contendo os dados de fechamento da cripto principal
                      e dos ativos auxiliares, unidos pela data.
    """
    dados_principal = obter_dados_crypto(cripto_principal, data_inicio, data_final)
    if not ativos_auxiliares:
        return dados_principal

    todos_dados = [dados_principal]
    for ativo in ativos_auxiliares:
        if ativo != cripto_principal:
            dados_aux = obter_dados_crypto(ativo, data_inicio, data_final)
            dados_aux = dados_aux[["Data", "Fechamento"]].rename(
                columns={"Fechamento": f"Fechamento_{ativo.replace(' ', '_')}"})
            todos_dados.append(dados_aux)

    df_final = todos_dados[0]
    for df in todos_dados[1:]:
        df_final = pd.merge(df_final, df, on="Data", how="inner")
    return df_final


def criar_features(df, lags, ma_windows):
    """Cria features de lag e médias móveis para o modelo."""
    df_feat = df.copy()
    df_feat['time_idx'] = np.arange(len(df_feat))

    cols_fechamento = [col for col in df_feat.columns if 'Fechamento' in col]

    for col in cols_fechamento:
        for lag in lags:
            df_feat[f'{col}_lag_{lag}'] = df_feat[col].shift(lag)
        for window in ma_windows:
            df_feat[f'{col}_ma_{window}'] = df_feat[col].rolling(window=window).mean()

    df_feat = df_feat.dropna()
    return df_feat

def prever_horizonte(model, dados, horizonte, scaler, lags, ma_windows, iniciar_hoje=False):
    """
    Gera previsões para um horizonte futuro de forma iterativa, onde cada nova
    previsão é usada para calcular as features do próximo passo.

    Args:
        model: O modelo de machine learning treinado.
        dados (pd.DataFrame): O DataFrame com os dados históricos.
        horizonte (int): O número de dias a serem previstos no futuro.
        scaler (StandardScaler): O scaler ajustado aos dados de treino.
        lags (list): A lista de lags usados como features.
        ma_windows (list): A lista de janelas de médias móveis usadas como features.
        iniciar_hoje (bool): Se True, as previsões começam a partir da data atual.

    Returns:
        np.array: Um array numpy com os valores previstos.
    """
    df_future = dados.copy()
    previsoes = []

    next_date = pd.Timestamp(datetime.date.today()) if iniciar_hoje else df_future['Data'].iloc[-1] + pd.Timedelta(days=1)

    for _ in range(horizonte):

        new_row = df_future.iloc[-1:].copy()
        new_row['Data'] = next_date

        df_temp = pd.concat([df_future, new_row], ignore_index=True)
        df_feat = criar_features(df_temp, lags, ma_windows)
        features_last = df_feat.drop(columns=['Data', 'Fechamento']).iloc[-1:]
        X_last = scaler.transform(features_last.values)
        pred = model.predict(X_last)[0]

        previsoes.append(pred)

        new_row['Fechamento'] = pred
        df_future = pd.concat([df_future, new_row], ignore_index=True)
        next_date += pd.Timedelta(days=1)

    return np.array(previsoes)


def preencher_gap_ate_hoje(model, dados, scaler, lags, ma_windows):
    """
    Preenche com previsões o intervalo de tempo entre a última data dos dados
    fornecidos e a data atual.

    Args:
        model: O modelo de machine learning treinado.
        dados (pd.DataFrame): O DataFrame com os dados históricos.
        scaler (StandardScaler): O scaler ajustado aos dados de treino.
        lags (list): A lista de lags usados como features.
        ma_windows (list): A lista de janelas de médias móveis usadas como features.

    Returns:
        tuple: Uma tupla contendo um DataFrame com as previsões do gap e o
               DataFrame original estendido com essas previsões.
    """
    df_future = dados.copy()
    gap_preds = []

    today = pd.Timestamp(datetime.date.today())
    last_date = df_future['Data'].iloc[-1]
    next_date = last_date + pd.Timedelta(days=1)

    while next_date < today:
        new_row = df_future.iloc[-1:].copy()
        new_row['Data'] = next_date

        df_temp = pd.concat([df_future, new_row], ignore_index=True)
        df_feat = criar_features(df_temp, lags, ma_windows)
        features_last = df_feat.drop(columns=['Data', 'Fechamento']).iloc[-1:]
        X_last = scaler.transform(features_last.values)
        pred = model.predict(X_last)[0]

        gap_preds.append({'Data': next_date, 'Fechamento': pred})

        new_row['Fechamento'] = pred
        df_future = pd.concat([df_future, new_row], ignore_index=True)
        next_date += pd.Timedelta(days=1)

    df_gap = pd.DataFrame(gap_preds)
    return df_gap, df_future


def highlight_fechamento(col):
    """
    Aplica uma cor ao texto com base na variação do valor em relação à linha anterior.
    Verde para aumento, vermelho para queda.

    Args:
        col (pd.Series): A coluna do DataFrame a ser estilizada.

    Returns:
        list: Uma lista de strings de estilo CSS.
    """
    diff = col.diff()
    colors = []
    for d in diff:
        if pd.isna(d):
            colors.append('')
        elif d >= 0:
            colors.append('color: green;')
        else:
            colors.append('color: red;')
    return colors


def preparar_resultados(cripto, dados_treino, y_pred_train, previsoes, mae, rmse, r2, gap_pred, horizonte, iniciar_hoje):
    """
    Organiza os dados de treino, previsões e métricas em um formato adequado
    para exibição no Streamlit.

    Args:
        cripto (str): Nome da criptomoeda.
        dados_treino (pd.DataFrame): Dados usados no treino.
        y_pred_train (np.array): Previsões do modelo sobre os dados de treino.
        previsoes (np.array): Previsões para o horizonte futuro.
        mae (float): Mean Absolute Error.
        rmse (float): Root Mean Squared Error.
        r2 (float): R-squared score.
        gap_pred (pd.DataFrame): Previsões para o gap até hoje.
        horizonte (int): Horizonte de previsão.
        iniciar_hoje (bool): Flag indicando se a previsão começa hoje.

    Returns:
        dict: Um dicionário contendo as métricas e os DataFrames para os gráficos.
    """
    start_date_prev = pd.Timestamp(datetime.date.today()) if iniciar_hoje else \
        dados_treino["Data"].iloc[-1] + pd.Timedelta(days=1)

    datas_prev = pd.date_range(start=start_date_prev, periods=horizonte, freq="D")
    df_prev = pd.DataFrame({"Data": datas_prev, "Fechamento": previsoes})

    chart_dados = pd.DataFrame({
        "Data": dados_treino["Data"].values,
        "Fechamento": dados_treino["Fechamento"].values.ravel(),
        "Tipo": "Histórico (Treino)",
    })

    chart_train = pd.DataFrame({
        "Data": dados_treino["Data"].values,
        "Fechamento": y_pred_train.ravel(),
        "Tipo": "Ajuste do Modelo",
    })

    chart_prev = df_prev.copy()
    chart_prev["Tipo"] = "Previsão"

    chart_gap = pd.DataFrame()
    if not gap_pred.empty:
        chart_gap = gap_pred.copy()
        chart_gap["Tipo"] = "Preenchimento"

    chart_df = pd.concat([chart_dados, chart_train, chart_gap, chart_prev], ignore_index=True)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "chart_df": chart_df,
        "df_prev": df_prev,
    }


def exibir_resultados(result, cripto, horizonte, prefix):
    """
    Renderiza os resultados (métricas, gráficos e tabela de previsões) na interface
    do Streamlit.

    Args:
        result (dict): O dicionário de resultados gerado por `preparar_resultados`.
        cripto (str): O nome da criptomoeda.
        horizonte (int): O horizonte de previsão.
        prefix (str): Um prefixo para as chaves dos widgets para evitar conflitos.
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Treino)", f"${result['mae']:,.2f}")
    col2.metric("RMSE (Treino)", f"${result['rmse']:,.2f}")
    col3.metric("R² (Treino)", f"{result['r2']:.4f}")

    st.line_chart(result['chart_df'], x="Data", y="Fechamento", color="Tipo")

    st.subheader(f"Previsão para os próximos {horizonte} dias:")
    st.dataframe(
        result['df_prev'].style
        .apply(highlight_fechamento, subset=['Fechamento'])
        .format({'Data': '{:%d/%m/%Y}', 'Fechamento': '${:,.2f}'})
        ,
        hide_index=True,
    )

    csv = result['chart_df'].to_csv(index=False).encode('utf-8')
    st.download_button(
        "Baixar dados como CSV", csv,
        f"previsao_{cripto.replace(' ', '_')}_{datetime.date.today().strftime('%Y%m%d')}_model{prefix}.csv",
        "text/csv", key=f'download-csv-{prefix}'
    )


def treinar_modelo(dados, ativos_auxiliares, horizonte, modelo, params, lags, ma_windows,iniciar_hoje=False):
    """
    Executa o pipeline completo de treinamento do modelo: cria features, escala os dados,
    treina o modelo, faz previsões e calcula as métricas de erro.

    Args:
        dados (pd.DataFrame): DataFrame com os dados de entrada.
        ativos_auxiliares (list): Lista de ativos auxiliares.
        horizonte (int): Número de dias para prever.
        modelo (str): Nome do modelo a ser treinado.
        params (tuple): Tupla com os hiperparâmetros do modelo.
        lags (list): Lista de lags para criar features.
        ma_windows (list): Lista de janelas de médias móveis.
        iniciar_hoje (bool): Se True, a previsão começa a partir de hoje.

    Returns:
        tuple: Contém as previsões, predições de treino, métricas (mae, rmse, r2),
               os dados com features e as previsões do gap.
    """
    dados_com_features = criar_features(dados, lags, ma_windows)

    features_df = dados_com_features.drop(columns=['Data', 'Fechamento'])
    X = features_df.values
    y = dados_com_features["Fechamento"].values


    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if modelo == "Regressão Linear":
        alpha, fit_intercept = params
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=42)
    elif modelo == "Random Forest":
        n_estimators, max_depth, min_samples_split = params
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, random_state=42
        )
    elif modelo == "Redes Neurais(MLP)":
        hidden_layer_sizes, activation, solver = params
        layers = tuple(int(h.strip()) for h in hidden_layer_sizes.split(',') if h.strip())
        model = MLPRegressor(
            hidden_layer_sizes=layers, activation=activation, solver=solver,
            max_iter=1000, random_state=42
        )
    elif modelo == "XGBoost":
        n_estimators, learning_rate, max_depth = params
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=42
        )
    elif modelo == "SVM":
        C, kernel, gamma = params
        model = SVR(C=C, kernel=kernel, gamma=gamma)
    else:
        return np.array([]), np.array([]), 0, 0, 0, pd.DataFrame()

    model.fit(X, y)
    y_pred_train = model.predict(X)
    mae = mean_absolute_error(y, y_pred_train)
    rmse = np.sqrt(mean_squared_error(y, y_pred_train))
    r2 = r2_score(y, y_pred_train)

    base_dados = dados.copy()
    df_gap = pd.DataFrame()
    if iniciar_hoje:
        df_gap, base_dados = preencher_gap_ate_hoje(model, base_dados, scaler, lags, ma_windows)

    previsoes = prever_horizonte(model, base_dados, horizonte, scaler, lags, ma_windows, iniciar_hoje)

    return previsoes, y_pred_train, mae, rmse, r2, dados_com_features, df_gap

def parametros_modelo(modelo, suffix=""):
    """
    Renderiza e retorna os widgets de hiperparâmetros para o modelo selecionado.

    Args:
        modelo (str): O nome do modelo selecionado.
        suffix (str): Um sufixo para as chaves dos widgets para permitir a comparação.

    Returns:
        tuple or None: Uma tupla com os valores dos hiperparâmetros selecionados.
    """
    if modelo == "Regressão Linear":
        return params_rl(suffix)
    elif modelo == "Random Forest":
        return params_rf(suffix)
    elif modelo == "Redes Neurais(MLP)":
        return params_mlp(suffix)
    elif modelo == "XGBoost":
        return params_gb(suffix)
    elif modelo == "SVM":
        return params_svm(suffix)
    return None


# --- Interface do Streamlit ---
st.title("Predição de Criptomoedas")

# Controle de estado para comparação de modelos
# Inicializa as variáveis de estado da sessão se ainda não existirem.
# Isso é crucial para manter o estado da aplicação entre as interações do usuário.
if 'run_train1' not in st.session_state:
    st.session_state.run_train1 = False
if 'run_train2' not in st.session_state:
    st.session_state.run_train2 = False
if 'show_compare' not in st.session_state:
    st.session_state.show_compare = False
if 'result_1' not in st.session_state:
    st.session_state.result_1 = None
if 'result_2' not in st.session_state:
    st.session_state.result_2 = None

# Coleta dos parâmetros de entrada do usuário através de widgets do Streamlit.
data_inicio = st.date_input("Qual a data de começo", value=None, min_value=datetime.date(2020, 1, 1),
                            max_value=datetime.date.today(), key='data_inicio')
if data_inicio:
    st.write("Data inicial da serie:", data_inicio)

data_final = st.date_input("Qual a data final", value=None, min_value=data_inicio, max_value=datetime.date.today(), key='data_final')
if data_final:
    st.write("Data final da serie:", data_final)

cripto = st.radio(
    "Qual criptomoeda deseja analisar",
    ["Bitcoin", "Ethereum", "Binance Coin", "Solana", "Cardano"],
    horizontal=True, index=None
)

ativos_auxiliares = st.multiselect(
    "Escolha os ativos auxiliares",
    [
        "Bitcoin",
        "Ethereum",
        "Binance Coin",
        "Solana",
        "Cardano",
        "S&P 500",
        "DXY",
        "Nasdaq 100",
        "Tesla",
        "Apple",
    ],
)

horizonte_pred = st.radio(
    "Qual o horizonte de previsão",
    ["1 dia", "3 dias", "5 dias", "7 dias"],
    horizontal=True, index=None
)

# Opção para o usuário decidir se a previsão começa após os dados de treino ou a partir de hoje.
inicio_prev = st.radio(
    "Aplicar previsões a partir de qual data?",
    ["Após dados de treino", "A partir de hoje"],
    horizontal=True
)
# st.caption("Selecione se as previsões começam logo após os dados de treino ou a partir de hoje.")

# --- MODIFICAÇÃO: Adicionados campos para definir o janelamento ---
st.write("### Parâmetros de Janelamento (Features)")
lags_input = st.text_input("Lags (valores separados por vírgula)", "1, 3, 5, 7")
ma_windows_input = st.text_input("Janelas das Médias Móveis (separadas por vírgula)", "5")
# --- Fim da Modificação ---

# Seleção do modelo de Machine Learning.
modelo_escolhido = st.radio(
    "Qual algoritmo de Aprendizado de Maquina deseja utilizar",
    ["Regressão Linear", "Random Forest", "Redes Neurais(MLP)", "XGBoost", "SVM"],
    horizontal=True, index=None
)


parametro = None
if modelo_escolhido:
    st.write("Você selecionou:", modelo_escolhido)
    # Obtém os hiperparâmetros para o modelo escolhido.
    parametro = parametros_modelo(modelo_escolhido)

# A lógica principal só é executada se todos os parâmetros necessários forem fornecidos.
if not (data_inicio and data_final and cripto and horizonte_pred and modelo_escolhido):
    st.write("Termine de preencher os parâmetros do modelo")
else:
    # Botão para iniciar o treinamento do primeiro modelo.
    # Clicar neste botão define 'run_train1' como True, disparando o processo de treinamento.
    if st.button('Treinar Modelo'):
        st.session_state.run_train1 = True
        st.session_state.result_1 = None # Reseta o resultado anterior

    # Bloco de código executado se o treinamento do modelo 1 foi iniciado.
    if st.session_state.run_train1:
        horizonte = int(horizonte_pred.split()[0])
        try:
            # Converte os inputs de lags e janelas para listas de inteiros.
            lags = [int(l.strip()) for l in lags_input.split(',') if l.strip()]
            ma_windows = [int(w.strip()) for w in ma_windows_input.split(',') if w.strip()]
        except ValueError:
            st.error("Por favor, insira apenas números inteiros separados por vírgula para os lags e médias móveis.")
            st.stop()

        # O treinamento só ocorre uma vez por clique, armazenando o resultado no estado da sessão.
        if st.session_state.result_1 is None:
            with st.spinner('Obtendo dados e treinando modelo...'):
                try:
                    dados = obter_dados_multiplos(cripto, ativos_auxiliares, data_inicio, data_final)
                    # Validação para garantir que há dados suficientes para as features de janela.
                    maior_janela = max(lags + ma_windows) if (lags or ma_windows) else 0
                    if len(dados) < maior_janela + 5:
                        st.error(
                            "Dados insuficientes para as janelas definidas. Selecione um período maior ou janelas menores.")
                        st.session_state.run_train1 = False
                    else:
                        # Chama a função principal de treinamento.
                        previsoes, y_pred_train, mae, rmse, r2, dados_treino, gap_pred = treinar_modelo(
                            dados, ativos_auxiliares, horizonte, modelo_escolhido, parametro, lags, ma_windows,
                            inicio_prev == "A partir de hoje"
                        )
                        # Prepara e armazena os resultados no estado da sessão.
                        st.session_state.result_1 = preparar_resultados(
                            cripto, dados_treino, y_pred_train, previsoes, mae, rmse, r2, gap_pred, horizonte,
                            inicio_prev == "A partir de hoje"
                        )
                except Exception as e:
                    st.error(f"Ocorreu um erro: {e}")
                    st.exception(e)
                    st.session_state.run_train1 = False

        # Se os resultados do modelo 1 estiverem prontos, exibe-os.
        if st.session_state.result_1:
            st.subheader(f"Previsão de preço para {cripto} - Modelo 1")
            exibir_resultados(st.session_state.result_1, cripto, horizonte, '1')
            # Botão para habilitar a interface de comparação.
            if st.button('Comparar Treino'):
                st.session_state.show_compare = True

        # Se a comparação foi habilitada, mostra a seção para o segundo modelo.
        if st.session_state.show_compare:
            st.write("## Segundo Modelo para Comparação")
            modelo2 = st.radio(
                "Escolha o segundo algoritmo",
                ["Regressão Linear", "Random Forest", "Redes Neurais(MLP)", "XGBoost", "SVM"],
                horizontal=True, key='modelo2'
            )
            parametro2 = parametros_modelo(modelo2,"_2") if modelo2 else None
            # Botão para treinar o segundo modelo.
            if st.button('Treinar Segundo Modelo'):
                st.session_state.run_train2 = True
                st.session_state.result_2 = None # Reseta o resultado anterior
                # Armazena os parâmetros do segundo modelo no estado para uso posterior.
                # st.session_state.modelo2 = modelo2
                st.session_state.parametro2 = parametro2

            # Bloco de código executado se o treinamento do modelo 2 foi iniciado.
            if st.session_state.run_train2:
                horizonte = int(horizonte_pred.split()[0])
                # O treinamento do segundo modelo também ocorre apenas uma vez.
                if st.session_state.result_2 is None:
                    with st.spinner('Treinando segundo modelo...'):
                        try:
                            dados = obter_dados_multiplos(cripto, ativos_auxiliares, data_inicio, data_final)
                            # Reutiliza os parâmetros do primeiro treino (datas, lags, etc.)
                            # mas com o novo modelo e seus hiperparâmetros.
                            previsoes, y_pred_train, mae, rmse, r2, dados_treino, gap_pred = treinar_modelo(
                                dados, ativos_auxiliares, horizonte, st.session_state.modelo2,
                                st.session_state.parametro2,
                                lags, ma_windows, inicio_prev == "A partir de hoje"
                            )
                            # Armazena os resultados do segundo modelo no estado da sessão.
                            st.session_state.result_2 = preparar_resultados(
                                cripto, dados_treino, y_pred_train, previsoes, mae, rmse, r2, gap_pred, horizonte,
                                inicio_prev == "A partir de hoje"
                            )
                        except Exception as e:
                            st.error(f"Ocorreu um erro: {e}")
                            st.exception(e)
                            st.session_state.run_train2 = False

                # Se os resultados do modelo 2 estiverem prontos, exibe-os.
                if st.session_state.result_2:
                    st.subheader(f"Previsão de preço para {cripto} - Modelo 2")
                    exibir_resultados(st.session_state.result_2, cripto, horizonte, '2')
                    # Botão para gerar e exibir o gráfico e a tabela de comparação.
                    if st.button('Comparar Resultados'):
                        res1 = st.session_state.result_1
                        res2 = st.session_state.result_2
                        # Prepara os dataframes para o gráfico de comparação.
                        hist = res1['chart_df'][res1['chart_df']['Tipo'] == 'Histórico (Treino)']
                        fit1 = res1['chart_df'][res1['chart_df']['Tipo'] == 'Ajuste do Modelo'].copy()
                        fit1['Tipo'] = 'Ajuste Modelo 1'
                        fit2 = res2['chart_df'][res2['chart_df']['Tipo'] == 'Ajuste do Modelo'].copy()
                        fit2['Tipo'] = 'Ajuste Modelo 2'
                        prev1 = res1['chart_df'][res1['chart_df']['Tipo'] == 'Previsão'].copy()
                        prev1['Tipo'] = 'Previsão Modelo 1'
                        prev2 = res2['chart_df'][res2['chart_df']['Tipo'] == 'Previsão'].copy()
                        prev2['Tipo'] = 'Previsão Modelo 2'
                        chart_comp = pd.concat([hist, fit1, fit2, prev1, prev2], ignore_index=True)
                        st.line_chart(chart_comp, x='Data', y='Fechamento', color='Tipo')

                        # Prepara a tabela de comparação das previsões.
                        df_comp = pd.merge(res1['df_prev'], res2['df_prev'], on='Data',
                                           suffixes=('_Modelo1', '_Modelo2'))

                        st.dataframe(
                            df_comp.style
                            .format(
                                {'Data': '{:%d/%m/%Y}', 'Fechamento_Modelo1': '${:,.2f}', 'Fechamento_Modelo2': '${:,.2f}'})
                            ,
                            hide_index=True
                        )