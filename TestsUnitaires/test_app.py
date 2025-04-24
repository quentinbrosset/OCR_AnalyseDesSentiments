import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
import logging
import os
import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    driver = None
    try:
        # Création du driver avec gestion explicite des erreurs
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Configuration des timeouts
        driver.set_page_load_timeout(30)  # 30 secondes max pour charger une page
        driver.implicitly_wait(10)  # Attente implicite de 10 secondes
        
        yield driver
    except WebDriverException as e:
        logger.error(f"Erreur WebDriver: {e}")
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la création du driver: {e}")
        raise
    finally:
        if driver:
            driver.quit()

def navigate_to_app(driver, url="http://localhost:8501"):
    """
    Méthode de navigation avec gestion des erreurs et logs
    """
    start_time = time.time()
    logger.info(f"Tentative de navigation vers {url}")
    
    try:
        driver.get(url)
        logger.info(f"Page chargée en {time.time() - start_time:.2f} secondes")
    except Exception as e:
        logger.error(f"Erreur de navigation : {e}")
        raise

def test_predict_positive_sentiment(driver):
    navigate_to_app(driver)
    
    try:
        # Attente et interaction avec le champ de texte
        tweet_input = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "textarea"))
        )
        tweet_input.clear()  # Nettoyer tout texte existant
        tweet_input.send_keys("I love this product!")
        
        # Attente et clic sur le bouton de prédiction
        predict_button = WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.XPATH, "//button[key='Prévoir le sentiment']"))
        )
        predict_button.click()
        
        # Vérification du résultat
        result_element = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.stMarkdown"))
        )
        result_text = result_element.text
        assert "Le sentiment prédictif est : Positif" in result_text, f"Résultat inattendu : {result_element.text}"
        
    except (TimeoutException, AssertionError) as e:
        logger.error(f"Échec du test de sentiment positif : {e}")
        driver.save_screenshot("positive_sentiment_error.png")
        raise

def test_predict_negative_sentiment(driver):
    navigate_to_app(driver)
    
    try:
        # Attente et interaction avec le champ de texte
        tweet_input = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "textarea"))
        )
        tweet_input.clear()  # Nettoyer tout texte existant
        tweet_input.send_keys("I hate this product!")
        
        # Attente et clic sur le bouton de prédiction
        predict_button = WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.XPATH, "//button[key='Prévoir le sentiment']"))
        )
        predict_button.click()
        
        # Vérification du résultat
        result_element = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.stMarkdown"))
        )
        result_text = result_element.text
        
        assert "Le sentiment prédictif est : Négatif" in result.text, f"Résultat inattendu : {result_element.text}"
        
    except (TimeoutException, AssertionError) as e:
        logger.error(f"Échec du test de sentiment négatif : {e}")
        driver.save_screenshot("negative_sentiment_error.png")
        raise

def test_invalid_input(driver):
    navigate_to_app(driver)
    
    try:
        # Attente et interaction avec le champ de texte
        tweet_input = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "textarea"))
        )
        tweet_input.clear()  # Nettoyer tout texte existant
        tweet_input.send_keys("  ")  # Entrer uniquement des espaces
        
        # Attente et clic sur le bouton de prédiction
        predict_button = WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@kind='secondary']"))
        )
        
        # Défiler jusqu'au bouton pour s'assurer qu'il est visible
        driver.execute_script("arguments[0].scrollIntoView(true);", predict_button)
        time.sleep(0.5)  # Petite pause pour s'assurer que le défilement est terminé
        
        # Clic sur le bouton avec JavaScript pour éviter les problèmes de clic
        driver.execute_script("arguments[0].click();", predict_button)
        
        # Petite pause pour laisser l'application réagir
        time.sleep(1)
        
        # Fonction de nouvelle tentative pour trouver le message d'avertissement
        def find_warning_message():
            warning_locators = [
                (By.CSS_SELECTOR, "div[data-testid='stAlert']"),
                (By.CSS_SELECTOR, "div.stAlert"),
                (By.XPATH, "//div[contains(text(), 'Veuillez entrer un tweet valide')]"),
                (By.XPATH, "//*[contains(text(), 'Veuillez entrer un tweet valide')]"),
                (By.XPATH, "//div[contains(@class, 'stAlert')]")
            ]
            
            for locator in warning_locators:
                elements = driver.find_elements(*locator)
                for element in elements:
                    if "Veuillez entrer un tweet valide" in element.text:
                        return element
            return None
        
        # Attente avec nouvelle tentative
        start_time = time.time()
        warning_message = None
        timeout = 30  # 30 secondes de timeout
        
        while time.time() - start_time < timeout:
            warning_message = find_warning_message()
            if warning_message:
                break
            time.sleep(0.5)  # Attendre 500ms avant de réessayer
            
        # Si aucun message n'est trouvé après le timeout
        if not warning_message:
            logger.error("Aucun message d'avertissement trouvé après le timeout")
            driver.save_screenshot("warning_not_found.png")
            print("URL actuelle:", driver.current_url)
            print("Page source:", driver.page_source)
            raise AssertionError("Impossible de trouver le message d'avertissement")
        
        # Vérifier le contenu du message
        assert "Veuillez entrer un tweet valide" in warning_message.text, \
            f"Message d'erreur inattendu : {warning_message.text}"
        
        logger.info("Test d'entrée invalide réussi : message d'avertissement correctement affiché")
        
    except (TimeoutException, AssertionError) as e:
        logger.error(f"Échec du test d'entrée invalide : {e}")
        driver.save_screenshot("invalid_input_error.png")
        raise
