from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from kmeans import kmeans_analizar_imagen
import os 
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TOKEN:
    raise ValueError("No se encontró la variable de entorno TELEGRAM_TOKEN")

async def say_hello(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hola! Soy tu bot K-means para imágenes.")

# NUEVO: Procesar fotos con K-means
async def process_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    photo_path = "temp_photo.jpg"
    await photo_file.download_to_drive(photo_path)
    
    # Aplicar K-means
    resultado_ruta, resumen = kmeans_analizar_imagen(photo_path, k=3)

    # Enviar imagen segmentada
    await update.message.reply_photo(photo=open(resultado_ruta, 'rb'))

    # Enviar resumen
    await update.message.reply_text(resumen)

# Crear el bot
application = ApplicationBuilder().token(TOKEN).build()
application.add_handler(CommandHandler("start", say_hello))
application.add_handler(MessageHandler(filters.PHOTO, process_photo))

# Ejecutar el bot
application.run_polling(allowed_updates=Update.ALL_TYPES)
