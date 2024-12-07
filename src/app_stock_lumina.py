import streamlit as st                  # pip install streamlit
import yfinance as yf                   # pip install yfinance
import pandas as pd
import numpy as np
import tensorflow as tf                 # pip install tensorflow 
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler  # pip install scikit-learn
import matplotlib.pyplot as plt         # pip install matplotlib
import plotly.graph_objects as plgo     # pip insall plotly

import time

BRANDING = """Stock LUMINA"""

MODEL = None


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
        layout="centered",
        initial_sidebar_state="expanded",
    )
    return



def get_stock_data(ticker: str, interval = "1d", _period = "") -> pd.DataFrame:
    max_period = {
        "1m": "max",
        "5m": "max",
        "15m": "max",
        "1h": "max",
        "1d": "10y",
        "5d": "10y",
        "1wk": "10y",
        "1mo": "10y",
        "3mo": "10y"
    }
    period = max_period[interval]
    if (_period):
        period = _period
    try:
        data: pd.DataFrame = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            # try with different period
            data = yf.download(ticker, interval=interval)
            if data.empty:
                st.error(f"""`[{ticker}] data not found! with interval: {interval}`""")
                return None
            
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
        if (isinstance(data.columns, pd.MultiIndex)):
            data.columns = [f"{col[0]}" for col in data.columns.to_flat_index()]
        return data

    except Exception as e:
        st.error(f"""`[{ticker}] data not found!\nPlease select another company.`""")
        st.warning(
            f"""
            ```py
            {e}
            ```
            """)
        return None


def prepare_data_for_prediction(stock_data, scaler, time_steps = 30):
    features = [
        "Close", 
        "MA_7", 
        "MA_14", 
        "MA_30", 
        "EMA_7", 
        "EMA_14", 
        "EMA_30", 
        "RSI", 
    ]
    # ensure the data has the required features
    stock_data = stock_data[features]
    
    # scale the data using same scaler
    scaled_data = scaler.transform(stock_data)
    
    x = []
    x.append(scaled_data[-time_steps:, :])
    return np.array(x)


def predict_stock_price(stock_data, model, scaler, time_steps=30):
    # Prepare the data for prediction using the last `time_steps` days
    prediction_input = stock_data[["Close", "MA_7", "MA_14", "MA_30", "EMA_7", "EMA_14", "EMA_30", "RSI"]].values[-time_steps:]
    
    # Scale the data using the same scaler that was used during training
    scaled_input = scaler.transform(prediction_input)
    
    # Reshape to match LSTM input shape (samples, time_steps, features)
    x_input = np.reshape(scaled_input, (1, time_steps, scaled_input.shape[1]))
    
    # Make prediction
    predicted_price = model.predict(x_input)
    
    # Inverse scale the predicted price (back to original scale)
    predicted_price = scaler.inverse_transform(np.hstack((predicted_price, np.zeros((predicted_price.shape[0], scaled_input.shape[1]-1)))))[:, 0]
    
    return predicted_price  # Return the predicted closing price (in original scale)



@st.cache_resource
def load_lstm_model(company: str):
    with st.spinner("Loading LSTM model.."):
        time.sleep(.2)
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
    
    # timeframes
    interval_mapping = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "1d": "1d",
        "5d": "5d",
        "1wk": "1wk",
        "1mo": "1mo",
        "3mo": "3mo"
    }
    timeframes = list(interval_mapping.keys())
    interval = interval_mapping["1d"]       # default interval
    
    # fetch stock data from yfinance
    stock_data = get_stock_data(ticker_company, interval=interval)
    
    
    # get lstm model
    try:
        global MODEL
        MODEL = load_lstm_model(ticker_company)
        # todo temporary display for 5 seconds
        success_message = st.sidebar.empty()
        success_message.success(f"`Model loaded successfully!`", icon="✅")
        time.sleep(3)
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
    st.title(f"""`{ticker_company}` ___~___ **{last_price_close:.2f} USD**""")
    st.write(f"""**{stock_data['Close'].index[0].date()}** to **{stock_data['Close'].index[-1].date()}**""")
    st.dataframe(stock_data)
    
    # interface
    st.markdown(f"<h3><span style='color: #f4ff33;'>{ticker_company}</span> Stock Data Visualization</h3>", unsafe_allow_html=True)
    selected_timeframe = st.segmented_control("Select the timeframe", timeframes, default="1d")
    new_interval = interval_mapping[selected_timeframe]

    if (interval != new_interval):
        interval = new_interval
        with st.spinner("Fetching stock data.."):
            time.sleep(.5)
            stock_data = get_stock_data(ticker_company, interval=interval)
    
    
    # Prediction Section
    st.markdown("Predict the Next Day's Closing Price")
    if st.button("Predict Next Day"):
        try:
            # Prepare data for prediction (last 30 days by default)
            last_days = 30
            st.markdown(stock_data)
            scaler = MinMaxScaler(feature_range=(0, 1))
            predicted_price = predict_stock_price(stock_data, MODEL, scaler, lastdays=last_days)
            st.success(f"Predicted Closing Price for {ticker_company} (Next Day): **{predicted_price[-1][0]:.2f} USD**")
        
        except Exception as e:
            st.error("Prediction failed!")
            st.warning(f"Error details: {e}")
    
    
    
    
    try:
        # Ensure stock data is valid
        if stock_data is None or stock_data.empty:
            st.write("Stock data not available right now!")
            st.error("Failed to fetch stock data!")
            return

        # Display the figure in Streamlit
        with st.spinner("Plotting the data.."):
            time.sleep(1.2)
            # Create the plotly figure
            fig = plgo.Figure()

            # Add line chart for closing price
            fig.add_trace(plgo.Scatter(
                x=stock_data.index, 
                y=stock_data["Close"], 
                mode="lines", 
                name="Close Price",
                yaxis="y1"  # Link to primary y-axis
            ))

            # Add bar chart for volume
            fig.add_trace(plgo.Bar(
                x=stock_data.index, 
                y=stock_data["Volume"], 
                name="Volume", 
                opacity=[.4, .2][timeframes.index(interval) > len(timeframes) * .7],
                marker=dict(color="yellow"),
                yaxis="y2"  # Link to secondary y-axis
            ))
            
            # Update layout to define dual axes
            fig.update_layout(
                title=f"{ticker_company} Stock",
                xaxis=dict(title="Date & Time"),
                yaxis=dict(
                    title="Price (USD)", 
                    titlefont=dict(color="blue"), 
                    tickfont=dict(color="blue")
                ),
                yaxis2=dict(
                    title="Volume (in Millions)", 
                    titlefont=dict(color="yellow"), 
                    tickfont=dict(color="yellow"),
                    anchor="x", 
                    overlaying="y", 
                    side="right"
                ),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                xaxis_rangeslider_visible=True,
                template="plotly_dark"  # Optional dark theme
            )

            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.write("Stock data not available right now!")
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