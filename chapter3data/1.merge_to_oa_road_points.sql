\timing

-- LOAD FILE TO BE MERGED
DROP TABLE merge.road_points_staging;
CREATE TABLE merge.road_points_staging(
    fid integer,
    cngmeters double precision,
    xcoord double precision, 
    ycoord double precision
);

copy merge.road_points_staging
FROM PROGRAM 'tail -n +2 /home/emily/phd/0_get_images/outputs/roads/gla_roads_points_20m.csv' DELIMITER as ',';

-- CREATE GEOGRAPHY
DROP TABLE merge.road_points;
CREATE TABLE merge.road_points(
    fid integer,
    geom geometry(Point, 4326),
    xcoord double precision, 
    ycoord double precision
);

INSERT INTO merge.road_points (fid, geom, xcoord, ycoord)
SELECT
    fid,
    ST_SetSRID(ST_MakePoint(xcoord, ycoord), 4326) as geom,
    xcoord,
    ycoord
FROM merge.road_points_staging;

-- DELETE INDEX idx_panoids_2011;
CREATE INDEX idx_road_points on merge.road_points USING gist(geom);

-- -- LOAD OUTPUT FILE
-- -- shp2pgsql -s 4326 OA_2011_London_gen_MHW.shp london postgres > oa.sql
-- -- psql -U emily -d london -f oa.sql

-- DELETE INDEX idx_oa;
-- CREATE INDEX idx_oa on poly USING gist(geom);


\COPY (SELECT points.fid as idx, points.xcoord AS xcoord, points.ycoord AS ycoord, oa.oa11cd AS oa_name FROM merge.road_points AS points  JOIN public.london AS oa ON ST_Contains(oa.geom, points.geom)) TO '/media/emily/south/chapter3data/outputs/road_points_merged_to_oa.csv' csv header;
