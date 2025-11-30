from app.db.crud import engine

try:
    with engine.connect() as conn:
        print("CONNECTED OK!!")
except Exception as e:
    print("ERROR:", e)
