import pymongo
from datetime import datetime

class ParkingDatabase:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["parking"]
        self.col = self.db["db_plat"]
        self.col.create_index([("plat_nomor", pymongo.ASCENDING), ("waktu_keluar", pymongo.ASCENDING)])

    def insert_entry(self, plat_nomor, waktu_masuk):
        try:
            existing_entry = self.col.find_one({"plat_nomor": plat_nomor, "waktu_keluar": None})
            if existing_entry:
                print(f"[INFO] Plat nomor {plat_nomor} sudah ada, tidak disimpan lagi.")
                return False
            self.col.insert_one({
                "plat_nomor": plat_nomor,
                "waktu_masuk": waktu_masuk,
                "waktu_keluar": None
            })
            print(f"[INFO] Plat nomor {plat_nomor} berhasil disimpan.")
            return True
        except Exception as e:
            print(f"[ERROR] Gagal insert entry: {e}")
            return False

    def update_exit(self, plat_nomor):
        try:
            result = self.col.update_one(
                {"plat_nomor": plat_nomor, "waktu_keluar": None},
                {"$set": {"waktu_keluar": datetime.now()}}
            )
            if result.modified_count > 0:
                print(f"[INFO] Plat nomor {plat_nomor} keluar, data diperbarui.")
                return True
            else:
                print(f"[WARNING] Plat nomor {plat_nomor} tidak ditemukan untuk update keluar.")
                return False
        except Exception as e:
            print(f"[ERROR] Gagal update exit: {e}")
            return False

    def fetch_all_entries(self):
        try:
            return list(self.col.find({}))
        except Exception as e:
            print(f"[ERROR] Gagal fetch entries: {e}")
            return []