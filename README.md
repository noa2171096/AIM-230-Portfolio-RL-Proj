# AIM-230-Portfolio-RL-Proj
## Abstract
Personal wealth management is a dominant market that is making use of the 
advancements of ML to deal with the vast quantities of data involved. The scope of the field 
ranges from the largest investment firms to the average person, each with varying needs. The 
central goal is optimizing wealth, whether in the short or long term. Part of a wealth portfolio can 
contain stocks, bonds, insurance, property, cryptocurrency, and standard income. 

The average person does not have the resources or financial literacy to 
properly engage in wealth management. Additionally, the sheer data that can drastically 
influence financial markets is overwhelming. An assistive, financially informed AI agent can 
address these problems and provide clients open-access to more financial freedom. 

In this project, a proof-of-concept RL model is deployed. This RL model is based on a shallow Deep Q-Network, where the model explores stock histories from Yfinance data and optimizes allocations of N assets (stocks) based on relevant features. These features include sentiment data, processed by FinBERT, moving averages, RSI, return from X days, etc. 

## Instructions

1. Clone this repo via "git clone ....". Navigate to root, PortfolioRL/
2. With Docker up and running in another window, apply "docker-compose up"
3. Download data, .pkl file, to data/, otherwise train model from scratch on specificed tickers
4. Go to http://localhost:8000/docs to access the FastAPI Swagger GUI and access model
5. Authentication and user verification is degraded at the moment
6. Go to "Analysis and ML Ops" to run model, modify the dict {
  "tickers": [
    "string"
  ],
  "horizon_days": 21,
  "episodes": 10,
  "capital": 10000,
  "force_retrain": false
}. For the time being, only tickers available are "NVDA", "GOOGL", "MSFT", and "AMZN", model training will be updated in near future...proof of concept for time being.
7. Retrieve recommendation by going to "/recommend/{job_id}" tab and entering the job ID in, the job ID is provided in step 6 response.
8. File upload is down for the moment...needs more coherent configuration. The idea is in the future user would be able to upload any relevant financial files, that can be organized and processed by an LLM. (more on future scope)

## Future Scope

The end goal is to have a comprehensive personal finance platform, that is both predictive and informative. The predictive element is handled by this RL DQN model, where the model develops an optimal strategy given a user's stock market interests. This can be fairly trivially extended to include other assets outside of stocks like savings, insurance, bonds, and real estate. The informative element will be handled by an outside LLM that can analyze personal finance documents and advise users. Additionally, it would be able to validate/rationalize the output of the RL model...
