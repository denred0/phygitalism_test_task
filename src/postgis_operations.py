import csv
import time
import pandas as pd
import plotly.express as px
import argparse
from typing import List
from tqdm import tqdm

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import Geometry, WKTElement
from sqlalchemy import Column, Integer
from sqlalchemy.orm import sessionmaker


def read_data(center_path: str,
              cluster_path: str) -> [List, List]:
    center_points = []
    with open(center_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in tqdm(spamreader, desc="Read center points data"):
            center_points.append(row)

    cluster_points = []
    with open(cluster_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in tqdm(spamreader, desc="Read cluster points data"):
            cluster_points.append(row)

    return center_points, cluster_points


def init_database(user: str,
                  password: str,
                  db: str):
    engine = create_engine(f'postgresql://{user}:{password}@localhost/{db}')
    Base = declarative_base()

    Session = sessionmaker(bind=engine)
    session = Session()

    class Center_points(Base):
        __tablename__ = 'center_points'

        id = Column(Integer, primary_key=True)
        point = Column(Geometry('POINTZ', dimension=3, srid=4326))

    class Cluster_points(Base):
        __tablename__ = 'cluster_points'

        id = Column(Integer, primary_key=True)
        center_id = Column(Integer)
        point_id = Column(Integer)
        point = Column(Geometry('POINTZ', dimension=3, srid=4326))
        color = Column(Geometry('POINTZ', dimension=3, srid=4326))

    return engine, session, Center_points, Cluster_points


def create_tables(engine, Center_points, Cluster_points):
    Center_points.__table__.create(engine)
    Cluster_points.__table__.create(engine)


def drop_tables(engine, Center_points, Cluster_points):
    Center_points.__table__.drop(engine)
    Cluster_points.__table__.drop(engine)


def save_center_points(center_points: List, Center_points, session):
    for p in tqdm(center_points, desc="Saving center points"):
        point_val = f"POINT({float(p[0])} {float(p[1])} {float(p[2])})"
        # point = Center_points(point=WKTElement(point_val, 4326))
        session.add(Center_points(point=point_val))
    session.commit()


def save_cluster_points(cluster_points: List, Cluster_points, session):
    for row in tqdm(cluster_points, desc="Saving cluster points"):
        point_val = f"POINT({float(row[2])} {float(row[3])} {float(row[4])})"
        # point = Cluster_points(point=WKTElement(point_val, 4326))

        color_val = f"POINT({float(row[5])} {float(row[6])} {float(row[7])})"
        # color = Cluster_points(color=WKTElement(color_val, 4326))
        session.add(Cluster_points(center_id=int(row[0]), point_id=int(row[1]), point=point_val, color=color_val))

    session.commit()


def select_points_from_bd(engine, center_id):
    with engine.connect() as con:
        rs = con.execute(
            # "SELECT ST_AsText(point) FROM center_points WHERE point=ST_MakePointM(221104.78766886334, 1906020.5446726466, 302.13441949)"
            # "SELECT ST_AsText(point) FROM center_points WHERE point=ST_AsEWKT(ST_MakePointM(220854.9513625, 1906057.10005188, 313.23361275), 4326)"
            # "SELECT ST_AsText(point) FROM center_points WHERE point=ST_AsEWKT( ST_MakePointM(-71.104, 42.315, 10));"
            # "SELECT ST_AsText(point) FROM center_points WHERE point=ST_GeomFromEWKT('POINT(220854.9513625 1906057.10005188 313.23361275)')"
            # "SELECT ST_AsText(point) FROM center_points WHERE point=ST_EWKTFromGeom('POINT(220854.9513625 1906057.10005188 313.23361275)')"
            # "SELECT ST_AsText(point) FROM center_points WHERE point=CAST(ST_Point(220854.9513625, 1906057.10005188, 313.23361275) AS geography)"

            f"SELECT ST_AsText(cl.point) FROM center_points cen INNER JOIN cluster_points cl on cen.id=cl.center_id WHERE cen.id={center_id}"
        )

        x = []
        y = []
        z = []
        for row in rs:
            values = row._data[0].replace('(', '').replace(')', '').split()
            x.append(float(values[2]))
            y.append(float(values[3]))
            z.append(float(values[4]))

        df = pd.DataFrame(list(zip(x, y, z)), columns=['X', 'Y', 'Z'])
        fig = px.scatter_3d(df, x='X', y='Y', z='Z')
        fig.show()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('center_path', type=str, help='Path to center_points data')
    parser.add_argument('cluster_path', type=str, help='Path to cluster_points data')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    center_path = opt.center_path
    cluster_path = opt.cluster_path

    # connection data
    user = "user"
    password = "267"
    db = "geo-db"
    engine, session, Center_points, Cluster_points = init_database(user, password, db)

    # read data from csv
    center_points, cluster_points = read_data(center_path, cluster_path)

    create_tables(engine, Center_points, Cluster_points)

    start_time_center_points = time.time()
    save_center_points(center_points, Center_points, session)
    print(f"Saving center points time: {round(time.time() - start_time_center_points, 1)} sec")

    start_time_cluster_points = time.time()
    save_cluster_points(cluster_points, Cluster_points, session)
    print(f"Saving cluster points time: {round(time.time() - start_time_center_points, 1)} sec")

    select_points_from_bd(engine, 1)

    drop_tables(engine, Center_points, Cluster_points)
