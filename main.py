__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from exec_streamlit.chatbox_gpt_light import main

if __name__ == "__main__":
    main()
