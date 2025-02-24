import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import os
from webdriver_manager.chrome import ChromeDriverManager

@pytest.fixture(scope="module")
def driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Désactiver pour mode non-headless
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    driver_path = ChromeDriverManager().install()
    driver = webdriver.Chrome(service=Service(driver_path), options=options)
    yield driver
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
