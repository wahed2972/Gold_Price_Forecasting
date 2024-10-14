from flask import Flask, render_template, request
import pandas as pd
import statsmodels.api as sm
from pickle import load
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load your dataset (df_price)
df_price = load(open('df_price.sav', 'rb'))

# Load the pre-trained ARIMA model
with open('arima_fit_final.sav', 'rb') as model_file:
    arima_fit_final = load(model_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    forecast_df = None  # Initialize forecast_df as None
    plot_url = None  # Initialize plot_url as None

    if request.method == 'POST':
        try:
            # Get the number of days input from the user
            periods = int(request.form['periods'])

            # Forecast for the specified number of days
            forecast = arima_fit_final.forecast(steps=periods)

            # Create DataFrame for forecast data
            datetime_index = pd.date_range(df_price.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
            forecast_df = pd.DataFrame(forecast.values, index=datetime_index, columns=['price'])
            forecast_df = forecast_df.round(2)

            # Plot the forecast data
            fig, ax = plt.subplots()
            ax.plot(df_price['price'][-200:], label='Historical Price')
            ax.plot(forecast_df['price'], "r--", label='Forecast')
            ax.set_title('Gold Price Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Gold Price')
            ax.legend(loc='upper left')

            # Convert plot to PNG image and encode it to display in HTML
            png_image = io.BytesIO()
            plt.savefig(png_image, format='png')
            png_image.seek(0)
            plot_url = base64.b64encode(png_image.getvalue()).decode('utf8')

        except Exception as e:
            print(f"Error occurred: {e}")

    return render_template('index.html', forecast=forecast_df, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
