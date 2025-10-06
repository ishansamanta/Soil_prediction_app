import os
os.environ['KIVY_WINDOW'] = 'mock'
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- Data Loading and Model Training ---
# This part remains the same. It trains the model when the app starts.
try:
    df = pd.read_csv('soil_dataset.csv')

    # Prepare the data
    X = df[['pH', 'Humidity (%)', 'Temperature (°C)']]
    y = df['Soil Type']

    # Encode the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    # Get the realistic range for the inputs from the dataset
    ph_min, ph_max = df['pH'].min(), df['pH'].max()
    humidity_min, humidity_max = df['Humidity (%)'].min(), df['Humidity (%)'].max()
    temp_min, temp_max = df['Temperature (°C)'].min(), df['Temperature (°C)'].max()

except FileNotFoundError:
    # If the CSV is not found, show an error in a popup and exit.
    def show_error_popup():
        popup = Popup(title="Fatal Error",
                      content=Label(text="soil_dataset.csv not found! Please make sure the dataset is in the same folder as the application."),
                      size_hint=(None, None), size=(400, 200))
        popup.open()
    show_error_popup()
    exit()

# --- Kivy App --- 

class SoilPredictorApp(App):

    def build(self):
        # Main Layout
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Title
        title = Label(text="M.O.O.N - Soil Predictor", font_size=30, size_hint_y=None, height=50)
        self.layout.add_widget(title)

        # Input Fields for pH, Humidity, and Temperature
        self.ph_input = self.create_input_field("Soil pH Level")
        self.humidity_input = self.create_input_field("Humidity (%)")
        self.temp_input = self.create_input_field("Temperature (°C)")
        
        # Add input fields to layout
        self.layout.add_widget(self.ph_input)
        self.layout.add_widget(self.humidity_input)
        self.layout.add_widget(self.temp_input)

        # Predict Button
        predict_button = Button(text="Predict Soil Type", size_hint_y=None, height=50, background_color=(0.2, 0.6, 1, 1))
        predict_button.bind(on_press=self.predict_soil_type)
        self.layout.add_widget(predict_button)

        # Status and Result Labels
        self.status_label = Label(text="", color=(1, 0, 0, 1), size_hint_y=None, height=30)
        self.result_label = Label(text="", color=(0, 1, 0, 1), size_hint_y=None, height=30)
        self.layout.add_widget(self.status_label)
        self.layout.add_widget(self.result_label)

        return self.layout

    def create_input_field(self, label_text):
        """Creates and returns a text input field with a label."""
        layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        label = Label(text=label_text, size_hint_x=None, width=150)
        text_input = TextInput(multiline=False, size_hint_x=0.6)
        layout.add_widget(label)
        layout.add_widget(text_input)
        return layout

    def predict_soil_type(self, instance):
        """Gets user input, validates it, and predicts the soil type."""
        try:
            # Retrieve values from entry fields
            ph = float(self.ph_input.children[0].text)
            humidity = float(self.humidity_input.children[0].text)
            temp = float(self.temp_input.children[0].text)

            # Validate input against the dataset's range
            if not (ph_min <= ph <= ph_max and
                    humidity_min <= humidity <= humidity_max and
                    temp_min <= temp <= temp_max):
                error_message = (f"Values are out of the realistic range.\n"
                                 f"Please use the following ranges:\n"
                                 f"pH: {ph_min:.2f}-{ph_max:.2f}, Humidity: {humidity_min:.2f}-{humidity_max:.2f}, Temp: {temp_min:.2f}-{temp_max:.2f}")
                self.status_label.text = error_message
                return

            # If validation passes, make the prediction
            input_data = np.array([[ph, humidity, temp]])
            prediction_encoded = model.predict(input_data)
            prediction = le.inverse_transform(prediction_encoded)

            # Update the result label with the prediction
            self.result_label.text = f"Predicted Soil Type: {prediction[0]}"

        except ValueError:
            self.status_label.text = "Please enter valid numerical values for all fields."
        except Exception as e:
            self.status_label.text = f"An unexpected error occurred: {e}"

if __name__ == '__main__':
    SoilPredictorApp().run()

