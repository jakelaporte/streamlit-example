# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:11:42 2023

@author: grover.laporte
"""
import skdh
import streamlit as st

file = st.file_uploader("Select the file to load:")
if file is not None:
  pipeline = skdh.Pipeline()
  pipeline.add(skdh.io.ReadBin())
  pipeline.add(skdh.preprocessing.CalibrateAccelerometer())
  pipeline.add(skdh.gait.Gait())
st.write("## Hello World")
