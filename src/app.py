import subprocess
import sys
try:
    import appdirs as ad
    ad.user_cache_dir = lambda *args: "/tmp"
    import streamlit as st                  # pip install streamlit
    import yfinance as yf                   # pip install yfinance
    import pandas as pd
    import numpy as np
    import tensorflow as tf                 # pip install tensorflow 
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler  # pip install scikit-learn
    import matplotlib.pyplot as plt         # pip install matplotlib
    import plotly.graph_objects as plgo     # pip insall plotly

except ImportError:
    # activate the conda venv 
    # subprocess.check_call([sys.executable, "conda", "activate", "lit"])  
    # install the required packages
    subprocess.check_call([sys.executable, "-m", "pip", "install", "appdirs"])
    import appdirs as ad
    ad.user_cache_dir = lambda *args: "/tmp"
    
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    
    import streamlit as st
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import plotly.graph_objects as plgo


import time

BRANDING = """Stock LUMINA"""


def pre_config() -> None:
    print("\nstreamlit version: ",st.__version__)
    
    city_sunrise_icon = ":city_sunrise:"
    graph_icon = "📈"
    # ref: https://fonts.google.com/icons?icon.query=graph
    graph_monitoring_google_icon = """<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#FFFF55"><path d="M160-120q-17 0-28.5-11.5T120-160v-40q0-17 11.5-28.5T160-240q17 0 28.5 11.5T200-200v40q0 17-11.5 28.5T160-120Zm160 0q-17 0-28.5-11.5T280-160v-220q0-17 11.5-28.5T320-420q17 0 28.5 11.5T360-380v220q0 17-11.5 28.5T320-120Zm160 0q-17 0-28.5-11.5T440-160v-140q0-17 11.5-28.5T480-340q17 0 28.5 11.5T520-300v140q0 17-11.5 28.5T480-120Zm160 0q-17 0-28.5-11.5T600-160v-200q0-17 11.5-28.5T640-400q17 0 28.5 11.5T680-360v200q0 17-11.5 28.5T640-120Zm160 0q-17 0-28.5-11.5T760-160v-360q0-17 11.5-28.5T800-560q17 0 28.5 11.5T840-520v360q0 17-11.5 28.5T800-120ZM560-481q-16 0-30.5-6T503-504L400-607 188-395q-12 12-28.5 11.5T131-396q-11-12-10.5-28.5T132-452l211-211q12-12 26.5-17.5T400-686q16 0 31 5.5t26 17.5l103 103 212-212q12-12 28.5-11.5T829-771q11 12 10.5 28.5T828-715L617-504q-11 11-26 17t-31 6Z"/></svg>"""
    analytics_google_icon = """<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#FFFF55"><path d="M280-280h80v-200h-80v200Zm320 0h80v-400h-80v400Zm-160 0h80v-120h-80v120Zm0-200h80v-80h-80v80ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm0-560v560-560Z"/></svg>"""
    
    st.set_page_config(
        page_title=f"{BRANDING} - AI Prediction",
        page_icon=graph_monitoring_google_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    return



def get_stock_data(ticker: str, period = "10y", interval = "1d") -> pd.DataFrame:
    day = "1d"
    month = "1mo"
    year = "1y"
    try:
        data: pd.DataFrame = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            st.error(f"""`[{ticker}] data not found! with period: {period} and interval: {interval}`""")
            return
        # """ example of multi-index columns from yfinance
        # MultiIndex([('Adj Close', 'AMZN'),
        #     (    'Close', 'AMZN'),
        #     (     'High', 'AMZN'),
        #     (      'Low', 'AMZN'),
        #     (     'Open', 'AMZN'),
        #     (   'Volume', 'AMZN')],
        #     names=['Price', 'Ticker'])
        # """
        # flatten the multi-index columns
        data.columns = [f"{col[0]}" for col in data.columns.to_flat_index()]

    except Exception as e:
        st.error(f"""`[{ticker}] data not found!\nPlease select another company.`""")
        st.warning(
            f"""
            ```py
            {e}
            ```
            """)
    return data



def predict_stock_price(stock_data, model, lastdays = 30):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(stock_data["Close"])
    
    # prepare the data for prediction (last N days)
    x_test = []
    for i in range(lastdays, len(data_scaled)):
        x_test.append(data_scaled[i-lastdays:i,0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # make the prediction
    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price



def load_lstm_model(company: str):
    model = load_model(f"./model/lumina_{company}.h5")
    return model



@st.dialog("Notify")
def notify():
    st.success("Model loaded successfully!")
    # reason = st.text_input("Because...")
    if st.button("Okay"):
        st.rerun()



def main() -> None:
    # st.title(BRANDING)
    st.markdown(f"<h1>Stock <span style='color: #f4ff33;'>LUMINA</span></h1>", unsafe_allow_html=True)
    st.write("This is a simple stock price prediction app using LSTM model.")
    
    # company selection
    st.sidebar.title("Company Selection")
    company_data = ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA", "NVDA"]
    
    ticker_company = st.sidebar.selectbox("Select the company", sorted(company_data))
    
    # fetch stock data from yfinance
    stock_data = get_stock_data(ticker_company)
    
    # get lstm model
    try:
        model = load_lstm_model(ticker_company)
        # todo temporary display for 5 seconds
        success_message = st.sidebar.empty()
        success_message.success(f"`Model loaded successfully!`", icon="✅")
        time.sleep(5)
        success_message.empty()
        
    except Exception as e:
        st.sidebar.error(f"""`[{ticker_company}] model not found!\nPlease select another company.`""")
        st.sidebar.warning(
            f"""
            ```py
            {e}
            ```
            """)    
    
    last_price_close = stock_data.iloc[-1]['Close']
    st.title(f"""`{ticker_company}` ___~___ **{last_price_close:.2f}**""")
    st.write(f"""**{stock_data['Close'].index[0].date()}** to **{stock_data['Close'].index[-1].date()}**""")
    st.write(stock_data)
    
    
    try:
        # plot the stock data into a graph
        fig = plgo.Figure()
        fig.add_trace(plgo.Scatter(
            x=stock_data.index, 
            y=stock_data["Close"],
            mode="lines", 
            name="Close Price"))
        
        fig.update_layout(
            title=f"Stock Price Graph",
            xaxis_title="Date & Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=True
        )
        
        st.markdown(f"<h3><span style='color: #f4ff33;'>{ticker_company}</span> Stock Data Visualization</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error("Failed to plot the data!")
        st.warning(
            f"""
            ```py
            {stock_data.columns}
            ```
            """)
        st.warning(
            f"""
            ```py
            {e}
            ```
            """)
    
    
    return



if __name__ == "__main__":
    pre_config()
    main()