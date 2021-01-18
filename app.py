import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from iso3166 import countries
from io import StringIO

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
     page_title="EDA Learning-by-Doing Web App",
     page_icon="",
     layout="wide",
     initial_sidebar_state="expanded",
)

DATA_URL = (
    "./covid19_tweets_cut.csv"
)

st.title("Exploratory Data Analysis of Tweets about Covid-19 🦠")
st.markdown("**Han Jiatong**  | School of Computing - NUS")


st.sidebar.title("Exploratory Data Analysis of Tweets on Covid-19 🦠")
st.sidebar.markdown("This application is a Streamlit dashboard used "
            "to observe the patterns of tweets")

with st.sidebar.beta_expander('Click to learn more about this dashboard'):
    st.markdown(
        """
    * The dataset is acquired from the kaggle website. Click [here](https://www.kaggle.com/gpreda/covid19-tweets) for more information. 
    * Sentiment analysis is intentionally omitted from this interactive notebook for better learning purpose.
    """)
with st.sidebar.beta_expander('references'):
    st.markdown(
        """
    * [Kaggle notebooks](https://www.kaggle.com/gpreda/covid19-tweets/notebooks?datasetId=798386&sortBy=voteCount)
    * [Coursera streamlit dashboard creating project](https://www.coursera.org/programs/national-university-of-singapore-on-coursera-bm9c5?currentTab=MY_COURSES&productId=7y4M746iEeqKwg4uzQo0NQ&productType=course&showMiniModal=true)
    """)

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data


data = load_data()

with st.beta_expander('First glimpse of dataset'):
    st.subheader('First glimpse of dataset')
    if st.checkbox('Show n random tweets',True,key='1'):
        number = st.slider('Number of tweets to take a look at:', 1, 10, 5,key='1')
        sample = data.sample(number)
        st.table(sample['text'])
        if st.button('Show raw data',key='raw'):
            st.write(sample)
with st.beta_expander('Data visualizations'):
    st.subheader("Visualizations")
    order = st.selectbox('Sort the users by (in ascending order)', ('number of tweets','number of followers', 'number of friends'),index=0,key='1')
    number = st.slider('Number of users to take a look at:', 10, 50, 30, key='2')
    toporbottom = st.radio('From the top or bottom?',('top','bottom'),index=0,key='1')
    visual = st.multiselect('Visualize the result by', ('bar chart','pie chart'),key='visual')

    ds = data['user_name'].value_counts().reset_index()
    ds.columns = ['user_name', 'tweets_count']
    ds = ds.sort_values(['tweets_count'])
    df = pd.merge(data, ds, on='user_name')

    if st.button("Visualize", key='sort_see'):
        if order == 'number of followers':
            df1 = df.sort_values('user_followers', ascending=toporbottom!='bottom')
            df1 = df1.drop_duplicates(subset='user_name', keep="first")
            df1 = df1[['user_name', 'user_followers', 'tweets_count']]
            df1 = df1.sort_values('user_followers', ascending=toporbottom!='bottom')
            if 'bar chart' in visual:
                fig = px.bar(
                    df1.tail(number), 
                    x="user_followers", 
                    y="user_name", 
                    color='tweets_count',
                    orientation='h', 
                    title=f'{toporbottom} {number} users by number of followers', 
                    width=800, 
                    height=800
                )
                st.plotly_chart(fig)
            if 'pie chart' in visual:
                fig = px.pie(
                    df1.tail(number),
                    values='user_followers',
                    names='user_name',
                    color='user_name',
                    title=f'{toporbottom} {number} users by number of followers', 
                    height=500
                )
                st.plotly_chart(fig)
        elif order == 'number of tweets':
            df2 = df.sort_values('tweets_count', ascending=toporbottom!='bottom')
            df2 = df2.drop_duplicates(subset='user_name', keep="first")
            df2 = df2[['user_name', 'user_followers', 'tweets_count']]
            df2 = df2.sort_values('tweets_count',ascending=toporbottom!='bottom')
            if 'bar chart' in visual:
                fig = px.bar(
                    df2.tail(number), 
                    x="tweets_count", 
                    y="user_name", 
                    color='user_followers',
                    orientation='h', 
                    title=f'{toporbottom} {number} users by number of tweets', 
                    width=800, 
                    height=800
                )
                st.plotly_chart(fig)
            if 'pie chart' in visual:
                fig = px.pie(
                    df2.tail(number),
                    values='tweets_count',
                    names='user_name',
                    color='user_name',
                    height=500
                )
                st.plotly_chart(fig)
        elif order == 'number of friends':
            df2 = df.sort_values('user_friends', ascending=toporbottom!='bottom')
            df2 = df2.drop_duplicates(subset='user_name', keep="first")
            df2 = df2[['user_name', 'user_friends', 'tweets_count']]
            df2 = df2.sort_values('user_friends',ascending=toporbottom!='bottom')
            if 'bar chart' in visual:
                fig = px.bar(
                    df2.tail(number), 
                    x="user_friends", 
                    y="user_name", 
                    color='tweets_count',
                    orientation='h', 
                    title=f'{toporbottom} {number} users by number of friends', 
                    width=800, 
                    height=800
                )
                st.plotly_chart(fig)
            if 'pie chart' in visual:
                fig = px.pie(
                    df2.tail(number),
                    values='user_friends',
                    names='user_name',
                    color='user_name',
                    height=500
                )
                st.plotly_chart(fig)
    if st.checkbox('Show source distribution for users',False):
        number = st.slider('Number of top sources to show',10,40,key='source')
        ds = df['source'].value_counts().reset_index()
        ds.columns = ['source', 'count']
        ds = ds.sort_values(['count'])

        fig = px.bar(
            ds.tail(number), 
            x="count", 
            y="source", 
            orientation='h', 
            title=f'Top {number} user sources by number of tweets', 
            width=800, 
            height=800
        )
        st.plotly_chart(fig)
with st.beta_expander('Time series analysis'):
        st.subheader("Time series analysis")

        df['date'] = pd.to_datetime(df['date']) 
        df = df.sort_values(['date'])
        df['day'] = df['date'].dt.day


        if st.checkbox('Show number of users tweeting per hour', False):
            group = st.radio('Group by',('day','hour','day and hour'),index=2,key='group')
            if group == 'day':
                df['day'] = df['date'].dt.day
                ds = df.groupby(['day'])['user_name'].count().reset_index()
                ds.columns = ['day', 'number_of_users']
                ds['day'] = 'Day' + ds['day'].astype(str)
                fig = px.line(ds, x="day", y="number_of_users", title='Number of unique users sending tweets per day')
                st.plotly_chart(fig)
            elif group == 'hour':
                df['hour'] = df['date'].dt.hour
                ds = df.groupby(['hour'])['user_name'].count().reset_index()
                ds.columns = ['hour', 'number_of_users']
                ds['day'] = 'Hour' + ds['hour'].astype(str)
                fig = px.line(ds, x="hour", y="number_of_users", title='Number of unique users sending tweets per hour')
                st.plotly_chart(fig)
            else:
                df['hour'] = df['date'].dt.hour
                df['date'] = df['date'].dt.date
                ds = df.groupby(['date','hour'])['user_name'].count().reset_index()
                ds.columns = ['date', 'hour', 'number_of_users']
                ds['hour_per_day'] = ds['date'].astype(str) +' at '+ ds['hour'].astype(str) + ':00'
                fig = px.line(ds, x="hour_per_day", y="number_of_users", title='Number of unique users sending tweets per hour per day')
                st.plotly_chart(fig)
with st.beta_expander('Geospatial analysis'):
    st.subheader('Geospatial analysis')
    df['location'] = df['user_location'].str.split(',', expand=True)[1].str.lstrip().str.rstrip()
    res = df.groupby(['day', 'location'])['text'].count().reset_index()

    country_dict = {}
    for c in countries:
        country_dict[c.name] = c.alpha3
        
    res['alpha3'] = res['location']
    res = res.replace({"alpha3": country_dict})

    country_list = ['England', 'United States', 'United Kingdom', 'London', 'UK']

    res = res[
        (res['alpha3'] == 'USA') | 
        (res['location'].isin(country_list)) | 
        (res['location'] != res['alpha3'])
    ]

    gbr = ['England', 'UK', 'London', 'United Kingdom']
    us = ['United States', 'NY', 'CA', 'GA']

    res = res[res['location'].notnull()]
    res.loc[res['location'].isin(gbr), 'alpha3'] = 'GBR'
    res.loc[res['location'].isin(us), 'alpha3'] = 'USA'
    res.loc[res['alpha3'] == 'USA', 'location'] = 'USA'
    res.loc[res['alpha3'] == 'GBR', 'location'] = 'United Kingdom'
    plot = res.groupby(['day', 'location', 'alpha3'])['text'].sum().reset_index()
    globe = st.radio('Plot the globe or selected countries?',('Globe','Selected countries'),index=0,key='globe')
    if globe == 'Globe':
        if st.button('Plot the map now!',key='map'):
            fig = px.choropleth(
                plot, 
                locations="alpha3",
                hover_name='location',
                color="text",
                animation_frame='day',
                projection="natural earth",
                color_continuous_scale=px.colors.sequential.Plasma,
                title='Tweets from different countries for every day',
                width=800, 
                height=600
            )
            st.plotly_chart(fig)
    else:
        locations = st.multiselect('Which country to visualize?',[country.name for country in countries])
        if st.button('Plot the map now!',key='map'):
            fig = px.choropleth(
                plot, 
                locations="alpha3",
                hover_name='location',
                color="text",
                animation_frame='day',
                projection="natural earth",
                color_continuous_scale=px.colors.sequential.Plasma,
                title='Tweets from different countries for every day',
                width=800, 
                height=600
            )
            st.plotly_chart(fig)
with st.beta_expander('Text analysis'):
    st.subheader('Text analysis')

    def build_wordcloud(df, title):
        wordcloud = WordCloud(
            background_color='gray', 
            stopwords=set(STOPWORDS), 
            max_words=50, 
            max_font_size=40, 
            random_state=666
        ).generate(str(df))

        fig = plt.figure(1, figsize=(14,14))
        plt.axis('off')
        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(top=2.3)

        plt.imshow(wordcloud)
        st.pyplot()


    ds = df.sort_values(['tweets_count'])
    countries = df.groupby(['user_location'])['text'].count().reset_index()
    countries.columns = ['location', 'count']
    countries = countries.sort_values(['count'],ascending=False)
    selected_country = st.selectbox('Select the locations you are interested in',[location for location in countries.location],key='select')
    ds = ds[ds['user_location'] == selected_country]
    if st.checkbox('Verified user only',False):
        ds = ds[ds['user_verified'] == 1]
    if st.checkbox('Retweeted user only',False):
        ds = ds[ds['is_retweet'] == 1]
    if ds.empty:
        e = RuntimeError('It seems that no entry survives after filtering process. Try other measures.')
        st.exception(e)
    else:
        clouds = st.slider(f'Select top {len(ds)} tweet users to visualize their collective word clouds',1,len(ds),key='select')
        cloud_result = st.radio('Build on tweet content or user desription?',('Content','User bio'),index=0,key='content')
        cloud_relation = {'Content':'text','User bio':'user_description'}
        ds = ds.tail(clouds)
        if st.button('Click to show the cloud',key='cloud'):
            build_wordcloud(ds[cloud_relation[cloud_result]], f'Word Cloud for top {clouds} users in {selected_country}')
with st.beta_expander('(Optional) Train the model'):
    st.subheader('(Optional) Train the model')
    st.markdown('With no labels for sentiments provided, this obviously calls for an effort specifically in the clustering process. For the moment kmeans clustering is adopted.')
    st.markdown('Now choose the k value:')
    k = st.slider('k would be',2,8,4,key='k')

    text = df.text.values

    @st.cache(persist=True,show_spinner=False)
    def vectorize(values):
        vec = TfidfVectorizer(stop_words="english")
        vec.fit(values)
        features = vec.transform(values)
        return features

    @st.cache(show_spinner=False)
    def fit(txt):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(vectorize(txt))
        res = kmeans.predict(vectorize(txt))
        return res

    if st.button('Start training',key='start_training'):
        df['Cluster'] = fit(text)
        if st.checkbox('Show representative word cloud for each cluster',True,key="represent"):
            k_show = st.slider('Which cluster?',0,k-1,key='cluster2')
            build_wordcloud(df[df['Cluster'] == k_show]['text'], f'Wordcloud for cluster {k_show}')
with st.beta_expander('Now to test your own data'):
    st.subheader('Now to test your own data')

    st.markdown('Upload your twitter text message and see if its sentiment is classified into which cluster')

    choice = st.radio('Choose your way of input text',('Type into the box below','Upload file'),index=0,key='choice')
    if choice == 'Type into the box below':
        st.markdown("#### Type in your twitter msg")
        _input = st.text_input('type here','Covid-19 impacts all our lives and profoundly changes the course of our progress at the prespective of humans',max_chars=200)
    else:
        newfile = st.file_uploader('Upload file here',['txt'])
        if newfile != None:
            bytesData = newfile.getvalue()
            encoding = 'utf-8'
            s=str(bytesData,encoding)
            _input = StringIO(s).read()

    printornot = st.checkbox('Print result',True)
    if st.button('Run classification',key='class'):
        new_text = np.append(text, _input)
        with st.spinner('Ready in less than a minute...'):
            result = fit(new_text)
        st.balloons()
        k = result[-1]
        if printornot:
            st.subheader(f'Your input text is classified into Cluster {k}')
            st.markdown("#### Some random examples from this cluster:")
            example = df[result[:-1] == result[-1]].sample(5).text.values
            st.table(example)
            st.markdown("#### And the word cloud for this cluster:")
            build_wordcloud(example, f'Wordcloud for cluster {k}')




