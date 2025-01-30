import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import os

@pytest.fixture(scope="module")
def driver():
    driver_path = os.path.join(os.path.dirname(__file__), "msedgedriver.exe")
    options = webdriver.EdgeOptions()
    # options.add_argument('--headless')  # Désactiver pour mode non-headless
    driver = webdriver.Edge(service=EdgeService(driver_path), options=options)
    yield driver
    driver.quit()

def test_predict_positive_sentiment(driver):
    driver.get("http://192.168.1.21:8501")
    
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

# Test de prédiction de sentiment négatif
def test_predict_negative_sentiment(driver):
    driver.get("http://192.168.1.21:8501")

    # Attendre que la zone de texte soit disponible
    tweet_input = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.TAG_NAME, "textarea"))
    )
    tweet_input.send_keys("I hate this product!")  # Entrer un tweet négatif

    # Attendre que le bouton soit cliquable, puis cliquer
    predict_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@kind='secondary']"))
    )
    predict_button.click()

    # Attendre que le résultat apparaisse
    result = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.stMarkdown"))
    ).text

    assert "Le sentiment prédictif est : Négatif" in result

# Test d'entrée invalide
def test_invalid_input(driver):
    driver.get("http://192.168.1.21:8501")

    # Attendre que la zone de texte soit disponible
    tweet_input = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.TAG_NAME, "textarea"))
    )
    tweet_input.send_keys("  ")  # Entrer uniquement des espaces

    # Attendre que le bouton soit cliquable, puis cliquer
    predict_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@kind='secondary']"))
    )
    predict_button.click()

    # Attendre que le message d'avertissement apparaisse
    warning_message = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.stAlert"))
    ).text
    assert "Veuillez entrer un tweet valide." in warning_message
