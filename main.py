from exec_streamlit.chatbox_gpt_light import main

import pysqlite3
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

if __name__ == "__main__":
    main()
