FROM python:3.8

# Create app directory
WORKDIR /app

# Install app dependencies
COPY /stock_data/alphavantage .

RUN pip install alphavantage

# Bundle app source
COPY src /app

EXPOSE 8080
CMD [ "python", "alpha_manager.py" ]