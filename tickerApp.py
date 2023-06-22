# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:08:03 2023

@author: grover.laporte
"""
import streamlit as st
import yfinance as yf
import pandas as pd

st.write("""
         ### Simple Stock Price Application
         Shown are the stock closing prices and volume of Google!
         
         """)
tickerSymbol = 'GOOGL'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period='1d',start='2010-5-31',
                              end='2020-5-31')
st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)