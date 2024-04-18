from collections import OrderedDict
import streamlit as st
import time

# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import streamlit_app.config as config

# TODO : you can (and should) rename and add tabs in the ./tabs folder, and import them here.
from streamlit_app.tabs import intro, radiocovid19, Data_visualisation, segmentation_mask, classification, references, conclusion_perspectives

# Initialization

# Détecte si la page active est la démo RadioCovid19
# Dans ce cas, le layout de la page doit changer
radiocovid_layout = False
if 'tabs_menu' in st.session_state:
    if st.session_state.tabs_menu == radiocovid19.sidebar_name:
        radiocovid_layout = True

if radiocovid_layout == True:
    st.config.set_option('theme.base' ,"dark")
    st.config.set_option('theme.textColor', "#FFFFFF")
    st.config.set_option('theme.backgroundColor', "#000000")
    st.config.set_option('theme.secondaryBackgroundColor', '#50B4C8')
    st.config.set_option('theme.primaryColor', '#50B4C8')
    st.set_page_config(
        page_title=config.TITLE,
        page_icon='🫁',
        layout='wide',
        initial_sidebar_state = "collapsed")
        
else:
    st.config.set_option('theme.base' ,"light")
    st.config.set_option('theme.textColor', "#000000")
    st.config.set_option('theme.backgroundColor', "#FFFFFF")
    st.config.set_option('theme.secondaryBackgroundColor', '#50B4C8')
    st.config.set_option('theme.primaryColor', '#4529DE')
    st.set_page_config(
        page_title=config.TITLE,
        page_icon='🫁',
        layout='centered',
        initial_sidebar_state = "expanded")

with open("streamlit_app/style.css", "r") as f:
    style = f.read()
st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (radiocovid19.sidebar_name, radiocovid19),
        (Data_visualisation.sidebar_name, Data_visualisation),
        (segmentation_mask.sidebar_name, segmentation_mask),
        (classification.sidebar_name, classification),
        (conclusion_perspectives.sidebar_name, conclusion_perspectives),
        (references.sidebar_name, references),
    ]
)


def onchange(**kwargs):
    st_write(**kwargs)

def run():

    if radiocovid_layout == True: 
        st.config.set_option('theme.base' ,"dark")
    else:
        st.config.set_option('theme.base' ,"light")

    st.sidebar.markdown('# RadioCovid19')
    
    st.sidebar.image(
        "streamlit_app/assets/poumon.jpg",
        use_column_width = True,
    )
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0, key='tabs_menu')
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")
    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]
    tab.run()


if __name__ == "__main__":
    run()

