'''
Selenium 활용하여 
visitjeju.net 의 여행지 url 전부 가져오는 스크립트. 
'''


from selenium import webdriver
from selenium.webdriver.common.by import By
import time 
from pprint import pprint

TOTAL_PAGE_COUNT = 125  # 스크롤할 페이지 수
LINE = 12 # 페이지 당 게시물 개수
CADENCE_TIME = 0.3 # 너무 빠르면 스크롤이 잘 이루어지지 않으므로 적당한 시간간격 선택.
urls = []


driver = webdriver.Chrome()

for page in range(1, TOTAL_PAGE_COUNT+1):
    driver.get(f"https://www.visitjeju.net/kr/detail/list?menuId=DOM_000001718000000000&cate1cd=cate0000000002#p{page}&pageSize=12&sortListType=reviewcnt&viewType=list&isShowBtag&tag")
    time.sleep(CADENCE_TIME)
    for l in range(1, LINE+1):
        try:
            urls.append( 
                driver.find_element(By.XPATH, f'//*[@id="content"]/div/div[2]/div[5]/ul/li[{l}]/dl/dt/a').get_attribute('href')
            )
        except:  # 게시물 안찾아지면 넘김
            print('no element found. skip this content.')

# print('##  total urls -- ')
# pprint(urls)

print('## url parsing finished. Saving urls...')

with open('visitjeju_places_url_list2.txt', 'w') as f:
    for u in urls:
        f.write(u + '\n')
print('## Saving url finished!')

driver.quit()
