import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import os
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

@pytest.fixture(scope="module")
def driver():
    # Configuration des options Chrome pour GitHub Actions
    chrome_options = webdriver.ChromeOptions() 
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--remote-debugging-port=9222')
    
    # Spécifier un répertoire utilisateur unique dans /tmp
    user_data_dir = '/tmp/chrome-user-data'
    os.makedirs(user_data_dir, exist_ok=True)
    chrome_options.add_argument(f'--user-data-dir={user_data_dir}')
    
    # Utilisation de webdriver_manager avec des paramètres explicites
    service = Service(ChromeDriverManager().install())
    
    try:
        # Création du driver avec gestion explicite des erreurs
        driver = webdriver.Chrome(service=service, options=chrome_options)
        yield driver
    except WebDriverException as e:
        print(f"Erreur WebDriver: {e}")
        raise
    except Exception as e:
        print(f"Erreur lors de la création du driver: {e}")
        raise
    finally:
        driver.quit()

def test_predict_positive_sentiment(driver):
    driver.get("http://192.168.1.14:8501")

    try:
        tweet_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "textarea"))
        )
        tweet_input.send_keys("I love this product!")

        predict_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@kind='secondary']"))
        )
        predict_button.click()

        result = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.stMarkdown"))
        ).text
        assert "Le sentiment prédictif est : Positif" in result

    except Exception as e:
        driver.save_screenshot("screenshot_error.png")
        raise

def test_predict_negative_sentiment(driver):
    driver.get("http://192.168.1.14:8501")

    tweet_input = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.TAG_NAME, "textarea"))
    )
    tweet_input.send_keys("I hate this product!")  # Entrer un tweet négatif

    predict_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@kind='secondary']"))
    )
    predict_button.click()

    result = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.stMarkdown"))
    ).text

    assert "Le sentiment prédictif est : Négatif" in result

def test_invalid_input(driver):
    driver.get("http://192.168.1.14:8501")

    tweet_input = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.TAG_NAME, "textarea"))
    )
    tweet_input.send_keys("  ")  # Entrer uniquement des espaces

    predict_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@kind='secondary']"))
    )
    predict_button.click()

    warning_message = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.stAlert"))
    ).text
    assert "Veuillez entrer un tweet valide." in warning_message
