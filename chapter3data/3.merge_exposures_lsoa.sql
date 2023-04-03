\timing

-- shp2pgsql -s 4326 LSOA_2011_London_gen_MHW.shp london postgres > lsoa.sql
-- -- psql -U emily -d london -f oa.sql

-- LOAD FILE TO BE MERGED
DROP TABLE merge.exposures_pm25_staging;
CREATE TABLE merge.exposures_pm25_staging(
    easting integer,
    northing integer,
    LAEI0808 double precision,
    LAEI0811 double precision,
    LAEI0815 double precision,
    lat double precision, 
    lon double precision
);

COPY merge.exposures_pm25_staging
FROM PROGRAM 'tail -n +2 /home/emily/phd/0_get_images/outputs/psql/exposures/LAEI08_NOXa_lat_lon.csv' DELIMITER as ',';

-- CREATE GEOGRAPHY
DROP TABLE merge.exposures_pm25;
CREATE TABLE merge.exposures_pm25(
    lat double precision, 
    lon double precision,
    geom geometry(Point, 4326),
    LAEI08 double precision,
    LAEI11 double precision,
    LAEI15 double precision
);

INSERT INTO merge.exposures_pm25 (lat, lon, geom, LAEI08, LAEI11, LAEI15)
SELECT
    lat,
    lon,
    -- geometry::Point(lat, lon, 4326) as geom,
    ST_SetSRID(ST_MakePoint(lon,lat), 4326) as geom,
    LAEI0808,
    LAEI0811,
    LAEI0815
FROM merge.exposures_pm25_staging;

-- DELETE INDEX idx_panoids_2011;
CREATE INDEX idx_exposures_pm25 on merge.exposures_pm25 USING gist(geom);

-- DELETE INDEX idx_oa;
CREATE INDEX idx_oa on poly USING gist(geom);


\COPY (SELECT points.lat AS lat, points.lon as lon, points.LAEI08 as LAEI08, points.LAEI11 as LAEI11, points.LAEI15 as LAEI15, oa.oa11cd AS oa_name FROM merge.exposures_pm25 AS points  JOIN public.london AS oa ON ST_Contains(oa.geom, points.geom)) TO '/home/emily/phd/drives/phd/chapter3data/outputs/exposure_NOXa_merge_oa.csv' csv header;

