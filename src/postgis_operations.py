import csv
import time
import pandas as pd
import plotly.express as px
import argparse
import pickle

from typing import List
from tqdm import tqdm

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import Geometry, WKTElement
from sqlalchemy import Column, Integer
from sqlalchemy.orm import sessionmaker


def read_data(centers_path: str,
              clusters_path: str,
              colors_path: str) -> [dict, dict]:
    with open(centers_path, 'rb') as f:
        center_points = pickle.load(f)

    with open(clusters_path, 'rb') as f:
        cluster_points = pickle.load(f)

    with open(colors_path, 'rb') as f:
        cluster_colors = pickle.load(f)

    return center_points, cluster_points, cluster_colors


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
        center_id = Column(Integer)
        point = Column(Geometry('POINTZ', dimension=3, srid=4326))

    class Cluster_points(Base):
        __tablename__ = 'cluster_points'

        id = Column(Integer, primary_key=True)
        center_id = Column(Integer)
        multipoint = Column(Geometry('MULTIPOINTZ', dimension=3, srid=4326))
        multicolor = Column(Geometry('MULTIPOINTZ', dimension=3, srid=4326))

    return engine, session, Center_points, Cluster_points


def create_tables(engine, Center_points, Cluster_points) -> None:
    Center_points.__table__.create(engine)
    Cluster_points.__table__.create(engine)


def drop_tables(engine, Center_points, Cluster_points) -> None:
    Center_points.__table__.drop(engine)
    Cluster_points.__table__.drop(engine)


def save_center_points(center_points: dict,
                       Center_points,
                       session,
                       srid: int) -> None:
    for key, p in tqdm(center_points.items(), desc="Saving center points"):
        point_val = f"POINT({float(p[0])} {float(p[1])} {float(p[2])})"
        point = Center_points(center_id=key, point=WKTElement(point_val, srid))
        session.add(point)
    session.commit()


def save_cluster_points(center_points: dict,
                        cluster_points: dict,
                        cluster_colors: dict,
                        Cluster_points,
                        session,
                        srid: int) -> None:
    for c_id in center_points.keys():
        # points
        points = cluster_points[c_id]
        multipoint_point = 'MULTIPOINT('
        for p in points:
            multipoint_point += str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + ', '

        multipoint_point = multipoint_point[:len(multipoint_point) - 2] + ')'

        # colors
        colors = cluster_colors[c_id]
        multipoint_color = 'MULTIPOINT('
        for c in colors:
            multipoint_color += str(c[0]) + ' ' + str(c[1]) + ' ' + str(c[2]) + ', '

        multipoint_color = multipoint_color[:len(multipoint_color) - 2] + ')'

        session.add(Cluster_points(
            center_id=c_id,
            multipoint=WKTElement(multipoint_point, srid),
            multicolor=WKTElement(multipoint_color, srid)))

    session.commit()


def select_points_from_bd(engine,
                          point: List,
                          srid: int) -> None:
    with engine.connect() as con:
        rs = con.execute(
            f"SELECT ST_AsText(cl.multipoint),  ST_AsText(cl.multicolor) FROM center_points cen "
            f"INNER JOIN cluster_points cl on cen.id=cl.center_id "
            f"WHERE cen.point=ST_GeomFromEWKT('SRID={srid};POINTZ({point[0]} {point[1]} {point[2]})')"
        )

        x = []
        y = []
        z = []
        colors_plot = []
        for row in rs:
            points = row._data[0].split("(")[1][:-1].split(',')
            colors = row._data[1].split("(")[1][:-1].split(',')
            for p, c in zip(points, colors):
                xyz = p.split()
                x.append(float(xyz[0]))
                y.append(float(xyz[1]))
                z.append(float(xyz[2]))

                rgb = c.split()
                color_hex = '#%02x%02x%02x' % (
                    int(float(rgb[0]) * 255), int(float(rgb[1]) * 255), int(float(rgb[2]) * 255))

                colors_plot.append(color_hex)

        df = pd.DataFrame(list(zip(x, y, z, colors_plot)), columns=['X', 'Y', 'Z', 'color'])
        fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='color')
        fig.show()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('center_path', type=str, help='Path to center_points data')
    parser.add_argument('cluster_path', type=str, help='Path to cluster_points data')
    parser.add_argument('colors_path', type=str, help='Path to points colors data')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    center_path = opt.center_path
    cluster_path = opt.cluster_path
    colors_path = opt.colors_path

    # connection data
    user = "user"
    password = "267"
    db = "geo-db"
    engine, session, Center_points, Cluster_points = init_database(user, password, db)

    # read data from csv
    center_points, cluster_points, cluster_colors = read_data(center_path, cluster_path, colors_path)

    create_tables(engine, Center_points, Cluster_points)

    srid = 4326

    start_time_center_points = time.time()
    save_center_points(center_points, Center_points, session, srid)
    print(f"Saving center points time: {round(time.time() - start_time_center_points, 1)} sec")

    start_time_cluster_points = time.time()
    save_cluster_points(center_points, cluster_points, cluster_colors, Cluster_points, session, srid)
    print(f"Saving cluster points time: {round(time.time() - start_time_cluster_points, 1)} sec")

    point = [220990.94025907823, 1906017.9723974576, 307.4823358869958]
    select_points_from_bd(engine, point, srid)

    drop_tables(engine, Center_points, Cluster_points)
