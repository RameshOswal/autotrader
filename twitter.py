from tweetfeels import TweetFeels
from threading import Thread
import time

lines_per_file  = 10
gap = 10

companies = ['apple', 'microsoft', 'google', 'facebook']
keywords = {}
keywords['apple'] = ['apple', 'appl']
keywords['microsoft'] = ['microsoft', 'msft']
keywords['google'] = ['google', 'googl']
keywords['facebook'] = ['facebook', 'fb']

def print_sentiments(feels, filename, seconds=10):
	file = open(filename+"_0"+".txt", "w") 
	iterations = 0
	while True:
		time.sleep(seconds)
		print(f'[{time.ctime()}] [{filename}] Sentiment Score: {feels.sentiment.value}')
		file.write(f'[{time.ctime()}] [{filename}] Sentiment Score: {feels.sentiment.value}\n')
		iterations = iterations + 1
		if (iterations % lines_per_file == 0):
			file.close()
			file = open(filename+"_"+ str(int(iterations/lines_per_file))+".txt", "w") 

def oath():
	consumer_key = 'JX0gSEugGSfQou3y5VFZBqSL3'
	consumer_secret = 'WnAPpDF1ywZrd00Yr9Y4hVQioOk7SmE8xuveKlGsAIRn4r7TIR'
	access_token = '134553563-yxyKvk4Qn6Vw6KPltVHvJYpLwhRYnT62xm6l0Dzd'
	access_token_secret = '5dC0hJWYQsZu79dNgBnzjSGPFT7K4Uaq00oLuWg6YsqQM'

	login = [consumer_key, consumer_secret, access_token, access_token_secret]
	return login

def main():
	login = oath()

	#for company in companies:
	company = 'microsoft'
	feels = TweetFeels(login, tracking=keywords[company])
	t = Thread(target=print_sentiments, args=[feels, company, gap])
	feels.start()
	t.start()

main()