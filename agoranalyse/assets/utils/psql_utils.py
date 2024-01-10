import psycopg2
from configparser import ConfigParser



def config(filepath='database.ini', section='postgresql'):
    """Read database configuration from file"""
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filepath)
    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filepath))
    return db


def get_connector(dbname: str, user: str, pwd: str):
    conn = psycopg2.connect(dbname=dbname, user=user, password=pwd)
    return conn


def get_connection(filepath: str="database.ini", section: str='postgresql') -> psycopg2.extensions.connection:
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config(filepath=filepath, section=section)

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
		
        # create a cursor
        cur = conn.cursor()
        
	# execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)
       
	# close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        print("Connexion Established to database.")
        if conn is not None:
            return conn


def generate_select_query(fields: list[str], table: str):
    query = "SELECT " + ", ".join(fields) + " FROM " + table
    return query


def get_responses(cursor: psycopg2.extensions.cursor):
    fields = ["text", "topicID", "probaTopic", "sentiment", "scoreSentiment", "frackingCount"]
    table = "Response"
    query = generate_select_query(fields, table)
    result = cursor.execute(query)
    return result


def generate_insert_query(fields: list[str], table: str, data):
    return
