# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:06:10 2023

@author: grover.laporte
"""
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.header("Adding distributions")
st.write("""Two discrete, uniform probability distribution added together
         gives a triangular distribution. This simple dice throwing 
         simulation shows this phenomenon. Start by adding one or two rolls
         per click and then add a large number.
         
         """)
stepsize = st.slider("Simulation speed (dice throws per click):",
                      min_value = 1,max_value = 500,step=1)
throws = 1
bar_width=30
incr = st.number_input("Click to throw more dice:",min_value = 0,
                       max_value = 10000, step = stepsize, format="%d")
np.random.randint(1,7,throws).reshape(-1,1)
red_die = pd.Series(np.random.randint(1,7,throws),name = "roll")
blue_die = pd.Series(np.random.randint(1,7,throws),name = "roll")
col1,col2,col3 = st.columns(3)

new_throws = np.random.randint(1,7,incr)
new_throws_blue = np.random.randint(1,7,incr)

incr_throws = pd.Series(new_throws,name = "roll")
incr_throws_blue = pd.Series(new_throws_blue,name="roll")

red_die = pd.concat([red_die,incr_throws],axis=0,ignore_index=True)
blue_die = pd.concat([blue_die,incr_throws_blue],axis=0,ignore_index=True)


red_distn = pd.DataFrame(red_die.value_counts().sort_index())
red_distn["distn"]=red_distn["count"]/red_distn["count"].sum()
red_distn["rolls"] = red_distn.index
c = alt.Chart(red_distn,title="Red Die").mark_bar(size=bar_width).encode(
    x="rolls",
    y="distn",
    color=alt.value("salmon"),
    )
col1.altair_chart(c,use_container_width=True)
col1.write(red_distn["distn"])

blue_distn = pd.DataFrame(blue_die.value_counts().sort_index())
blue_distn["distn"]=blue_distn["count"]/blue_distn["count"].sum()
blue_distn["rolls"] = blue_distn.index
c = alt.Chart(blue_distn,title="Blue Die").mark_bar(size=bar_width).encode(
    x="rolls",
    y="distn",
    color=alt.value("cadetblue"),
    )
col2.altair_chart(c,use_container_width=True)
col2.write(blue_distn["distn"])

dice_tot = red_die+blue_die
dice_hist = pd.DataFrame(dice_tot.value_counts().sort_index())
dice_hist.columns = ["total"]
dice_hist["distn"] = dice_hist["total"]/dice_hist["total"].sum()
dice_hist["rolls"] = dice_hist.index

c = alt.Chart(dice_hist,title="Red + Blue").mark_bar(size=bar_width).encode(
    x="rolls",
    y="distn",
    color=alt.value("mediumpurple"),
    )
col3.altair_chart(c,use_container_width=True)
col3.write(dice_hist["distn"])