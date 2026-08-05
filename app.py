"""
Smart Irrigation and Water Consumption Prediction System
Flask Web Server Application
"""

import os
import datetime
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, jsonify, Response
)

from database.db import (
    init_db, create_user, verify_user, reset_password,
    save_prediction, get_all_predictions, get_prediction_stats
)

from predict import (
    predict_water_requirement, forecast_7day_consumption,
    get_loaded_models
)

from utils.recommendation import generate_recommendations
from utils.weather import fetch_live_weather
from utils.exporter import export_to_csv, export_to_excel, export_to_pdf
from utils.chatbot import get_chatbot_response

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'smart_irrigation_secret_key_2026_x89a')

# Initialize DB on startup
with app.app_context():
    init_db()

def is_logged_in():
    return 'user_id' in session

@app.context_processor
def inject_user():
    return {
        'logged_in': is_logged_in(),
        'current_username': session.get('username', ''),
        'current_email': session.get('email', '')
    }

# ==================== PUBLIC & AUTH ROUTES ====================

@app.route('/')
def index():
    stats = get_prediction_stats()
    live_weather = fetch_live_weather()
    return render_template('index.html', stats=stats, weather=live_weather)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_logged_in():
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username_email = request.form.get('username_email', '').strip()
        password = request.form.get('password', '').strip()

        user = verify_user(username_email, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['email'] = user['email']
            flash(f"Welcome back, {user['username']}!", 'success')
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials! Please check your username/email and password.", 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if is_logged_in():
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        if password != confirm_password:
            flash("Passwords do not match!", 'warning')
            return render_template('register.html')

        success, result = create_user(username, email, password)
        if success:
            flash("Registration successful! Please log in with your credentials.", 'success')
            return redirect(url_for('login'))
        else:
            flash(result, 'danger')

    return render_template('register.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_pwd():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        new_password = request.form.get('new_password', '').strip()

        success, msg = reset_password(email, new_password)
        if success:
            flash(msg, 'success')
            return redirect(url_for('login'))
        else:
            flash(msg, 'danger')

    return render_template('forgot_password.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out safely.", 'info')
    return redirect(url_for('login'))

# ==================== CORE APPLICATION ROUTES ====================

@app.route('/dashboard')
def dashboard():
    if not is_logged_in():
        flash("Please log in to access the Dashboard.", 'warning')
        return redirect(url_for('login'))

    stats = get_prediction_stats()
    recent_predictions = get_all_predictions()[:5]
    weather = fetch_live_weather()

    # Get baseline sample forecast
    sample_input = {
        'Temperature': weather['temperature'],
        'Humidity': weather['humidity'],
        'Soil_Moisture': weather['soil_moisture'],
        'Rainfall': weather['rainfall'],
        'Wind_Speed': weather['wind_speed'],
        'Solar_Radiation': weather['solar_radiation'],
        'Soil_Type': 'Loamy',
        'Crop_Type': 'Wheat',
        'Crop_Growth_Stage': 'Development',
        'Previous_Water_Consumption': 180.0,
        'Day_Of_Year': datetime.datetime.now().timetuple().tm_yday
    }
    forecast_data = forecast_7day_consumption(sample_input)

    # ML model metrics info
    irrigation_meta, _ = get_loaded_models()
    model_metrics = irrigation_meta.get('metrics', {})

    return render_template(
        'dashboard.html',
        stats=stats,
        recent_predictions=recent_predictions,
        weather=weather,
        forecast=forecast_data,
        model_metrics=model_metrics,
        best_model_name=irrigation_meta.get('model_name', 'Random Forest'),
        best_r2=irrigation_meta.get('best_r2', 0.94)
    )

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if not is_logged_in():
        flash("Please log in to make predictions.", 'warning')
        return redirect(url_for('login'))

    prediction_result = None
    recommendations = []
    forecast_info = None

    if request.method == 'POST':
        farmer_name = request.form.get('farmer_name', session.get('username', 'Farmer')).strip()
        date_str = request.form.get('date', datetime.date.today().strftime('%Y-%m-%d'))
        
        input_data = {
            'Temperature': float(request.form.get('temperature', 28.0)),
            'Humidity': float(request.form.get('humidity', 55.0)),
            'Soil_Moisture': float(request.form.get('soil_moisture', 35.0)),
            'Rainfall': float(request.form.get('rainfall', 0.0)),
            'Wind_Speed': float(request.form.get('wind_speed', 8.0)),
            'Solar_Radiation': float(request.form.get('solar_radiation', 6.5)),
            'Soil_Type': request.form.get('soil_type', 'Loamy'),
            'Crop_Type': request.form.get('crop_type', 'Wheat'),
            'Crop_Growth_Stage': request.form.get('growth_stage', 'Development'),
            'Previous_Water_Consumption': float(request.form.get('prev_consumption', 180.0)),
            'Day_Of_Year': datetime.datetime.strptime(date_str, '%Y-%m-%d').timetuple().tm_yday if date_str else 180
        }

        # Run Prediction Engine
        prediction_result = predict_water_requirement(input_data)
        recommendations = generate_recommendations(input_data, prediction_result)
        forecast_info = forecast_7day_consumption(input_data)

        # Save to SQLite Database
        db_record = {
            'user_id': session.get('user_id', 1),
            'farmer_name': farmer_name,
            'date': date_str,
            'temp': input_data['Temperature'],
            'humidity': input_data['Humidity'],
            'soil_moisture': input_data['Soil_Moisture'],
            'rainfall': input_data['Rainfall'],
            'wind_speed': input_data['Wind_Speed'],
            'solar_rad': input_data['Solar_Radiation'],
            'soil_type': input_data['Soil_Type'],
            'crop_type': input_data['Crop_Type'],
            'growth_stage': input_data['Crop_Growth_Stage'],
            'prev_consumption': input_data['Previous_Water_Consumption'],
            'water_required': prediction_result['water_required_liters'],
            'duration_mins': prediction_result['irrigation_duration_mins'],
            'status': prediction_result['status'],
            'confidence_score': prediction_result['confidence_score'],
            'model_used': prediction_result['model_used'],
            'recommendation': "; ".join(recommendations[:2])
        }

        save_prediction(db_record)
        flash("Water requirement prediction calculated & saved to history!", 'success')

    weather = fetch_live_weather()
    return render_template(
        'prediction.html',
        result=prediction_result,
        recommendations=recommendations,
        forecast=forecast_info,
        weather=weather,
        today_date=datetime.date.today().strftime('%Y-%m-%d')
    )

@app.route('/forecast')
def forecast_view():
    if not is_logged_in():
        flash("Please log in to view water forecasts.", 'warning')
        return redirect(url_for('login'))

    weather = fetch_live_weather()
    sample_input = {
        'Temperature': weather['temperature'],
        'Humidity': weather['humidity'],
        'Soil_Moisture': weather['soil_moisture'],
        'Rainfall': weather['rainfall'],
        'Wind_Speed': weather['wind_speed'],
        'Solar_Radiation': weather['solar_radiation'],
        'Soil_Type': 'Loamy',
        'Crop_Type': 'Wheat',
        'Crop_Growth_Stage': 'Development',
        'Previous_Water_Consumption': 180.0,
        'Day_Of_Year': datetime.datetime.now().timetuple().tm_yday
    }
    forecast_data = forecast_7day_consumption(sample_input)
    return render_template('forecast.html', forecast=forecast_data, weather=weather)

@app.route('/history')
def history():
    if not is_logged_in():
        flash("Please log in to access history.", 'warning')
        return redirect(url_for('login'))

    search_query = request.args.get('search', '').strip()
    crop_filter = request.args.get('crop', 'All').strip()

    predictions = get_all_predictions(search_query=search_query, crop_filter=crop_filter)
    return render_template(
        'history.html',
        predictions=predictions,
        search_query=search_query,
        selected_crop=crop_filter
    )

# ==================== EXPORT ROUTES ====================

@app.route('/export/<format_type>')
def export_history(format_type):
    if not is_logged_in():
        flash("Authentication required for export.", 'danger')
        return redirect(url_for('login'))

    predictions = get_all_predictions()

    if format_type == 'csv':
        csv_data = export_to_csv(predictions)
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=Smart_Irrigation_History.csv"}
        )
    elif format_type == 'excel':
        excel_data = export_to_excel(predictions)
        return Response(
            excel_data,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-disposition": "attachment; filename=Smart_Irrigation_History.xlsx"}
        )
    elif format_type == 'pdf':
        pdf_data = export_to_pdf(predictions)
        return Response(
            pdf_data,
            mimetype="application/pdf",
            headers={"Content-disposition": "attachment; filename=Smart_Irrigation_History.pdf"}
        )
    else:
        flash("Invalid export format requested.", 'warning')
        return redirect(url_for('history'))

# ==================== INFO & CONTACT ROUTES ====================

@app.route('/about')
def about():
    irrigation_meta, _ = get_loaded_models()
    return render_template(
        'about.html',
        metrics=irrigation_meta.get('metrics', {}),
        best_model=irrigation_meta.get('model_name', 'Random Forest'),
        best_r2=irrigation_meta.get('best_r2', 0.94),
        best_rmse=irrigation_meta.get('best_rmse', 12.5)
    )

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name', '')
        email = request.form.get('email', '')
        message = request.form.get('message', '')
        flash(f"Thank you, {name}! Your message has been received. Our agricultural support team will reply to {email} shortly.", 'success')
        return redirect(url_for('contact'))

    return render_template('contact.html')

# ==================== API ENDPOINTS ====================

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json() or {}
    user_msg = data.get('message', '')
    bot_reply = get_chatbot_response(user_msg)
    return jsonify({'response': bot_reply})

@app.route('/api/weather')
def api_weather():
    lat = request.args.get('lat', 20.5937)
    lon = request.args.get('lon', 78.9629)
    city = request.args.get('city', 'Field Station')
    data = fetch_live_weather(lat=float(lat), lon=float(lon), city=city)
    return jsonify(data)

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', error="404 - Page Not Found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('index.html', error="500 - Internal Server Error"), 500

if __name__ == '__main__':
    print("\nStarting Smart Irrigation & Water Consumption Prediction Flask App...")
    app.run(host='0.0.0.0', port=5000, debug=True)
