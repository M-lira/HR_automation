import pandas as pd
import os
import smtplib
import ssl
import logging
import time
import schedule
from datetime import date
from threading import Lock, Thread
from fpdf import FPDF, XPos, YPos
from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from flask import Flask, request, jsonify, render_template, redirect, url_for, abort, send_file
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

#::::::::::::::::::::::::::::....DATABASE....::::::::::::::::::::::::::::#

DATABASE_URL = "sqlite:///./db/Hr_automation.db"
EMPLOYEES_PATH = './data/Employees.csv'
NEEDING_TRAINING_FOLDER = './reports/Needing-Training'


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelo de empregado (Employee)
class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, unique=True, index=True)
    name = Column(String)
    department = Column(String)
    start_date = Column(Date)
    termination_date = Column(Date, nullable=True)
    last_training_date = Column(Date, nullable=True)
    email = Column(String)


# Modelo das formações (Training)
class Training(Base):
    __tablename__ = 'trainings'

    id = Column(Integer, primary_key=True, index=True)
    available_training = Column(String, unique=True)

# Criar tabelas
Base.metadata.create_all(bind=engine)
#........................................................................#


#::::::::::::::::::::::::::::....CHATGROQ....::::::::::::::::::::::::::::#

GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

chat_groq = ChatGroq(api_key=GROQ_API_KEY, temperature=0, model="llama3-70b-8192")

#.........................................................................#


#::::::::::::::::::::::::::::....FUNCTIONS....::::::::::::::::::::::::::::#

# Gera as sugestões de formação baseada no departamento utilizando Groq
def generate_training_suggestions(department, training_options):
    trainings_str = "\n".join(training_options)
    
    prompt = f"""
    A IA deverá começar a responder sem saudação inicial.

    A IA deverá responder na língua em que foi feita a pergunta. 

    Por favor, responde apenas em português de Portugal (nunca português do Brasil) e justifica resposta com base no seu conhecimento e de forma sucinta.

    A IA é prestável e informativa, fornecendo uma resposta apenas com base nos documentos fornecidos.

    Se a IA não tiver informações ou se as informações não estiverem presentes nos documentos fornecidos, a IA declara claramente que não sabe ou que não tem acesso a essas informações.

    Se a IA não souber a resposta a uma pergunta, ela diz a verdade e não inventa uma resposta.

    Dada a seguinte lista de opções de formação disponíveis: 
    
    {trainings_str}
    
    Para '{department}', sugira as 3 opções de formação mais relevantes e benéficas para o funcionário.
    """

    logging.info(f"Generated prompt: {prompt}")

    try:
        response = chat_groq.predict(prompt)
        logging.info(f"Groq API response: {response}")  
        return response.strip()
    except Exception as e:
        logging.error(f"Error generating suggestions: {e}")
        return "Error generating suggestions"


# Lê as opções de formação do ficheiro csv
def read_training_options_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=';')
        return df['Available training'].tolist()
    except Exception as e:
        logging.error("Error reading training options from CSV: %s", e)
        return []


def read_employees_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=';')

        value_to_check = pd.Timestamp(date.today().year, 1, 1)
        df['Last Training Date'] = pd.to_datetime(df['Last Training Date'], errors='coerce')
        filter_mask = df['Last Training Date'] < value_to_check
        filtered_df = df[filter_mask]
        return filtered_df
    except Exception as e:
        logging.error("Error reading training options from CSV: %s", e)
        return []


# Vai buscar as formações à bd ou ao ficheiro csv
def fetch_available_trainings(session, csv_file_path=None):
    if csv_file_path and os.path.exists(csv_file_path):
        return read_training_options_from_csv(csv_file_path)
    
    trainings = session.query(Training.available_training).all()
    return [training[0] for training in trainings]


# Função para sugerir as formações
def suggest_trainings_for_employees(employee_df, session, training_options_csv_file):
    training_options = fetch_available_trainings(session, training_options_csv_file)
    suggestions = {}
    for _, employee in employee_df.iterrows():
        department = employee['Department']
        suggestion = generate_training_suggestions(department, training_options)
        suggestions[employee['Employee ID']] = {
            'name': employee['Name'],
            'suggestions': suggestion
        }
        time.sleep(3)
    return suggestions


def process_and_generate_report():
    try:
        session = SessionLocal()

        today = datetime.now().date()
        one_year_ago = today - timedelta(days=365)
        date_folder = today.strftime("%Y-%m-%d")

        # Create directory for report files
        report_dir = os.path.join('./reports', date_folder)
        os.makedirs(report_dir, exist_ok=True)

        # Debugging: Log the directory creation
        logging.info(f"Report directory: {report_dir}")
        employees_need_training = read_employees_from_csv(EMPLOYEES_PATH)
        
        if employees_need_training.size > 0:
            # Convert to DataFrame
            df = pd.DataFrame(employees_need_training)

            # Include the current date in the filenames
            date_str = today.strftime("%Y-%m-%d")
            
            filtered_csv_file = os.path.join(NEEDING_TRAINING_FOLDER, f'{date_str}_Filtered_Employees_Needing_Training.csv')
            df.to_csv(filtered_csv_file, index=False, sep=';')

            # Debugging: Log CSV creation
            logging.info(f"Filtered employees saved to: {filtered_csv_file}")
            
            # Training options from CSV or DB
            training_options_csv_file = './data/Trainings.csv'
            training_suggestions = suggest_trainings_for_employees(df, session, training_options_csv_file)
            suggestions_df = pd.DataFrame.from_dict(training_suggestions, orient='index').reset_index()

            # Renaming columns, debug step
            suggestions_df.rename(columns={'index': 'Employee ID'}, inplace=True)
            logging.info(f"Suggestions DataFrame created with columns: {suggestions_df.columns}")

            suggestions_csv_file = os.path.join(report_dir, f'{date_str}_Employee_Training_Suggestions.csv')
            suggestions_df.to_csv(suggestions_csv_file, sep=';')

            # Debugging: Log suggestions CSV creation
            logging.info(f"Training suggestions saved to: {suggestions_csv_file}")

            # Generate PDFs
            pdf_file_employees = os.path.join(report_dir, f'{date_str}_Employees_Training_Report.pdf')
            pdf_file_suggestions = os.path.join(report_dir, f'{date_str}_Employee_Training_Suggestions_Report.pdf')
            
            # Debugging PDF generation steps
            logging.info(f"Generating PDF for employees: {pdf_file_employees}")
            generate_pdf_employees(filtered_csv_file, pdf_file_employees)

            logging.info(f"Generating PDF for suggestions: {pdf_file_suggestions}")
            generate_pdf_suggestions(suggestions_csv_file, pdf_file_suggestions)

            sender_email = "noneexample8@gmail.com"  # Use your email
            sender_password = "kgqd uqzk vbcp isyi"  # App-specific password or regular password
            receiver_email = "noneexample8@gmail.com"
            subject = f"Relatório {date_str} | Formação obrigatória de colaboradores "
            body = "Em anexo, envio o relatório semanal com a lista de colaboradores que, no último ano, não completaram as 40 horas de formação obrigatória, conforme estipulado no artigo 131º do Código do Trabalho."
            attachments = [pdf_file_employees, pdf_file_suggestions]

            # Call the function to send the email with attachments
            send_email_with_attachments(sender_email, sender_password, receiver_email, subject, body, attachments)

            logging.info("Email with report sent successfully.")

        else:
            logging.info("No employees need training based on the last training date.")
            
    except Exception as e:
        logging.error(f"Error processing and generating report: {e}")


def generate_pdf_employees(csv_file, pdf_file):
   
    try:
        df = pd.read_csv(csv_file, delimiter=';')

        # Debugging
        print("Employees DataFrame columns:", df.columns.tolist())

        # Remove os espaços a mais
        df.columns = df.columns.str.strip()

        # Verifica se as colunas existem
        required_columns = ['Employee ID', 'Name', 'Department', 'Last Training Date']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        if df.empty:
            logging.warning("CSV file is empty. No PDF will be generated.")
            return

        pdf = FPDF()
        pdf.add_page()
        pdf.add_font('OpenSans', '', 'fonts/Open_Sans/static/OpenSans-Regular.ttf')
        pdf.add_font('OpenSansBold', '', 'fonts/Open_Sans/static/OpenSans-Bold.ttf')

        pdf.set_font("OpenSansBold", size=20)
        pdf.cell(0, 10, "Relatório de Formação Obrigatória", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)

        pdf.set_font("OpenSans", size=10)
        page_width = pdf.w - 2 * pdf.l_margin
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(page_width, 10, (
            "Lista de colaboradores que, no último ano, não completaram as 40 horas de formação obrigatória, conforme estipulado no artigo 131.º do Código do Trabalho."
        ), align='L')
        pdf.ln(10)

        col_widths = [30, 50, 50, 50]
        headers = ["Employee ID", "Name", "Department", "Last Training Date"]

        total_width = sum(col_widths)
        x_start = (page_width - total_width) / 2 + pdf.l_margin

        # Titulos da tabela
        pdf.set_font("OpenSansBold", size=12)
        pdf.set_x(x_start)
        for header, width in zip(headers, col_widths):
            pdf.cell(width, 10, header, border=1, align='C')
        pdf.ln()

        # Linhas da tabela
        pdf.set_font("OpenSans", size=12)
        for _, row in df.iterrows():
            pdf.set_x(x_start)
            for header in headers:
                col_index = headers.index(header)
                pdf.cell(col_widths[col_index], 10, str(row.get(header, 'N/A')), border=1, align='C')
            pdf.ln()

        # Guardar pdf
        print(f"Saving PDF file: {pdf_file}")  
        pdf.output(pdf_file)
        logging.info(f"PDF generated successfully: {pdf_file}")

    except Exception as e:
        logging.error(f"Error generating PDF from CSV: {e}")
        raise


def generate_pdf_suggestions(csv_file, pdf_file):
    try:
        # Read data from CSV file
        df = pd.read_csv(csv_file, delimiter=';')

        # Print column names for debugging
        print("Suggestions DataFrame columns:", df.columns.tolist())

        # Strip extra spaces from column names
        df.columns = df.columns.str.strip()

        # Check if required columns exist (excluding 'department' as it may be optional)
        required_columns = ['Employee ID', 'name', 'suggestions']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check if 'department' exists, handle if missing
        department_exists = 'department' in df.columns

        if df.empty:
            logging.warning("CSV file is empty. No PDF will be generated.")
            return

        pdf = FPDF()
        pdf.add_page()
        pdf.add_font('OpenSans', '', 'fonts/Open_Sans/static/OpenSans-Regular.ttf')
        pdf.add_font('OpenSansBold', '', 'fonts/Open_Sans/static/OpenSans-Bold.ttf')

        pdf.set_font("OpenSansBold", size=20)
        pdf.cell(0, 10, "Sugestões de Formação", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)

        pdf.set_font("OpenSans", size=12)
        page_width = pdf.w - 2 * pdf.l_margin
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(page_width, 10, (
            "Sugestões de formação para os colaboradores com base no departamento em que trabalham."
        ), align='L')
        pdf.ln(10)

        # Loop through each row in the DataFrame
        for _, row in df.iterrows():
            # Print Employee ID, Name, and Department if it exists
            pdf.set_font("OpenSansBold", size=12)
            pdf.set_x(pdf.l_margin)
            if department_exists:
                department = row.get('Department', '')
                pdf.cell(0, 10, f"ID {row.get('Employee ID', '')} | Nome: {row.get('name', '')} | Departamento: {department}", ln=True)
            else:
                pdf.cell(0, 10, f"ID {row.get('Employee ID', '')} | Nome: {row.get('name', '')}", ln=True)

            # Clean and format suggestions, switching back to regular font
            pdf.set_font("OpenSans", size=12)
            suggestions = row.get('suggestions', '')
            cleaned_suggestions = suggestions.replace('*', '').strip()
            pdf.set_x(pdf.l_margin)  
            pdf.multi_cell(0, 10, cleaned_suggestions)
            pdf.ln(10)  

        # Save the PDF
        print(f"Saving PDF file: {pdf_file}") 
        pdf.output(pdf_file)
        logging.info(f"PDF generated successfully: {pdf_file}")

    except Exception as e:
        logging.error(f"Error generating PDF from CSV: {e}")
        raise


def send_email_with_attachments(sender_email, sender_password, receiver_email, subject, body, attachments):
    try:
        logging.basicConfig(level=logging.INFO)

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        for file_path in attachments:

          if not os.path.exists(file_path):
                logging.error(f"Attachment {file_path} does not exist.")
                continue

          try:
                part = MIMEBase('application', 'octet-stream')
                with open(file_path, 'rb') as attachment_file:
                    part.set_payload(attachment_file.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition', 
                    f'attachment; filename={os.path.basename(file_path)}'
                )
                msg.attach(part)
                
                logging.info(f"Attached file: {file_path}")

          except Exception as e:
                logging.error(f"Failed to attach {file_path}: {e}")
        
        context = ssl.create_default_context()

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            logging.info("Connecting to the email server...")
            server.login(sender_email, sender_password)
            logging.info("Login successful.")
            server.sendmail(sender_email, receiver_email, msg.as_string())

        logging.info(f"Email sent successfully to {receiver_email}")

    except smtplib.SMTPAuthenticationError:
        logging.error("Authentication failed. Check your email and password.")
    except smtplib.SMTPRecipientsRefused:
        logging.error("Recipient refused. Check the receiver's email address.")
    except Exception as e:
        logging.error(f"Error sending email: {e}")
        raise

scheduled_job = None
job_lock = Lock()  


# Background thread to run scheduled jobs
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

def job():
    logging.info("Scheduled job has started.")
    suggestions = process_and_generate_report()
    if suggestions:
        logging.info(f"Suggestions generated: {suggestions}")
    else:
        logging.info("No suggestions were generated.")

scheduler_thread = Thread(target=run_scheduler)
scheduler_thread.daemon = False
scheduler_thread.start()

#........................................................................#


#:::::::::::::::::::::::::::::....ROUTES....:::::::::::::::::::::::::::::#


#página inicial
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/report-files', methods=['GET'])
def get_report_files():
     try:
        today = datetime.now().date()
        today_str = today.strftime("%Y-%m-%d")
        reports_dir = './reports'
        report_data = {}

        for folder in os.listdir(reports_dir):
            folder_path = os.path.join(reports_dir, folder)
            if os.path.isdir(folder_path):
                report_data[folder] = []
                for file in os.listdir(folder_path):
                    if file.endswith('.pdf'):
                        report_data[folder].append(file)

        return render_template('reports.html', reports=report_data)

     except Exception as e:
        logging.error(f"Error listing reports: {e}")
        return "An error occurred while listing reports.", 500


@app.route('/start-job', methods=['GET'])
def start_job():
    global scheduled_job
    try:
        with job_lock:
            if scheduled_job is None:
                scheduled_job = schedule.every().day.at("09:00").do(job)
                #scheduled_job = schedule.every(5).seconds.do(job)
                logging.info("Job started and scheduled.")
                return render_template('start_job.html')
            else:
                logging.warning("Job is already running.")
                return render_template('start_job.html')
    except Exception as e:
        logging.error(f"Failed to start job: {str(e)}")
        return jsonify({"error": f"Failed to start job: {str(e)}"}), 500


@app.route('/needing-training-history', methods=['GET'])
def needing_training_history():
    try:
        file_contents = {}
        for root, dirs, files in os.walk(NEEDING_TRAINING_FOLDER):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path, delimiter=';')
                    file_contents[file] = df.to_dict(orient='records')

        # Render the HTML template with the data
        return render_template('needing_training_history.html', data=file_contents), 200
        
    except Exception as e:
        logging.error(f"Failed: {str(e)}")
        return jsonify({"error": f"Failed: {str(e)}"}), 500


employees_df = pd.read_csv('./data/Employees.csv', delimiter=';')
available_training_df = pd.read_csv('./data/Trainings.csv', delimiter=';')


@app.route('/employees')
def employees():
    # Clean column names
    employees_df.columns = employees_df.columns.str.strip()
    employees = employees_df.to_dict(orient='records')  
    return render_template('employees.html', employees=employees)


@app.route('/trainings')
def trainings():
    # Clean column names
    available_training_df.columns = available_training_df.columns.str.strip()
    trainings = available_training_df['Available training'].dropna().tolist()  # Extract training categories
    return render_template('trainings.html', trainings=trainings)


@app.route('/files')
def list_files_html():  
    # Logic to list files (e.g., from a directory)
    files = os.listdir(NEEDING_TRAINING_FOLDER)  
    return render_template('list_files.html', files=files)


@app.route('/list-files', methods=['GET'])
def list_files():
    files = os.listdir('./reports/Needing-Training')
    return render_template('list_files.html', files=files)


@app.route('/get-file/<filename>', methods=['GET'])
def get_file(filename):
    file_path = f'./reports/Needing-Training/{filename}'
    try:
        with open(file_path, 'r') as file:
            content = file.read() 
        return render_template('file_content.html', filename=filename, content=content)  # Pass content to template
    except FileNotFoundError:
        abort(404)


@app.route('/reports')
def reports():
    return render_template('start_job.html')


@app.route('/generate-report',methods=['GET'])
def generate_report():
    try:
        process_and_generate_report()
        return jsonify({"message": "Relatório gerado e enviado com sucesso!"}), 200
    except Exception as e:
        logging.error(f"Error in generating report: {e}")
        return jsonify({"error": "Falha ao gerar o relatório."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5003)