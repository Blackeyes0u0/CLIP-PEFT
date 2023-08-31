import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_image_title(url):
    # 웹 드라이버 초기화
    driver = webdriver.Chrome()
    driver.implicitly_wait(5) # or bigger second
    # 열고자 하는 채널 -> 동영상 목록으로 된 url 페이지를 엶
    driver.get(url)
    image_list = list() # 썸네일을 받을 수 있는 주소 저장용 리스트
    title_list = list() # 썸네일 제목을 저장하는 리스트
    view_list = list() # 조회수 리스트
    periods_list = list()
    idx = 1
    while True:
        try:
            img_xpath = '/html/body/ytd-app/div/ytd-page-manager/ytd-browse/ytd-two-column-browse-results-renderer/div[1]/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-grid-renderer/div[1]/ytd-grid-video-renderer['+str(idx)+']/div[1]/ytd-thumbnail/a/yt-img-shadow/img'
            title_xpath = '/html/body/ytd-app/div/ytd-page-manager/ytd-browse/ytd-two-column-browse-results-renderer/div[1]/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-grid-renderer/div[1]/ytd-grid-video-renderer['+str(idx)+']/div[1]/div[1]/div[1]/h3/a'
            viewcnt_xpath = '/html/body/ytd-app/div/ytd-page-manager/ytd-browse/ytd-two-column-browse-results-renderer/div[1]/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-grid-renderer/div[1]/ytd-grid-video-renderer['+str(idx)+']/div[1]/div[1]/div[1]/div[1]/div[1]/div[2]/span[1]'
            period_xpath = '/html/body/ytd-app/div/ytd-page-manager/ytd-browse/ytd-two-column-browse-results-renderer/div[1]/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-grid-renderer/div[1]/ytd-grid-video-renderer['+str(idx)+']/div[1]/div[1]/div[1]/div[1]/div[1]/div[2]/span[2]'
            # 이미지가 곧바로 로드 되지 않을 때, 20초간 강제로 기다림
            img = WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.XPATH, img_xpath)))
            if img is None:
                print(idx, 'img is not loaded.')

            # 한 페이지에 약 8개 불러오는 데, 동영상 목록을 추가 불러오기 위해 스크롤 내림
            if idx % 8 == 0 :
                driver.execute_script('window.scrollBy(0, 1080);')
                time.sleep(2)
                driver.execute_script('window.scrollBy(0, 1080);')
                time.sleep(2)
                driver.execute_script('window.scrollBy(0, 1080);')
                time.sleep(2)

            # 썸네일 주소를 리스트에 저장
            # image = driver.find_element_by_xpath(img_xpath)
            image = driver.driver.find_element(By.XPATH,img_xpath)
            img_url = image.get_attribute('src')
            image_list.append(img_url)

            # 타이틀을 리스트에 저장
            # title = driver.find_element_by_xpath(title_xpath)
            title = driver.driver.find_element(By.XPATH,title_xpath)
            title_list.append(title.text)

            # 조회수를 리스트에 저장
            # view = driver.find_element_by_xpath(viewcnt_xpath)
            view = driver.driver.find_element(By.XPATH,viewcnt_xpath)
            view_list.append(view.text)

            # 시간을 리스트에 저장
            # period = driver.find_element_by_xpath(period_xpath)
            period = driver.driver.find_element(By.XPATH,period_xpath)
            periods_list.append(period)
            print(idx, title.text, view.text, period.text, img_url)

            idx += 1

        except Exception as e:
            print()
            print(e)
            break
    assert len(image_list) == len(title_list)
    # driver.close()
    return image_list, view_list, periods_list, title_list

# 자이언트 펭TV
url1 = 'https://www.youtube.com/channel/UCtckgmUcpzqGnzcs7xEqMzQ/videos'
image1, view1, period1, title1 = get_image_title(url1)