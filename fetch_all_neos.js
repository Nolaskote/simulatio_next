// fetch_all_neos.js
import fs from "fs";
import https from "https";

const API_KEY = "FgQdqLRDLps8aBoZyeM9zSPKcyObPvbDcXomch5k";
const PAGE_SIZE = 20;
const BASE_URL = `https://api.nasa.gov/neo/rest/v1/neo/browse?api_key=${API_KEY}&size=${PAGE_SIZE}`;
const RETRY_LIMIT = 5;
const PAUSE_MS = 1200; // 1.2 segundos entre peticiones

const path = "./public/data/neos.json";

// Lee el archivo existente si existe
let existingNeos = [];
let startPage = 0;
if (fs.existsSync(path)) {
  const file = fs.readFileSync(path, "utf-8");
  existingNeos = JSON.parse(file);
  startPage = Math.floor(existingNeos.length / PAGE_SIZE);
  console.log(`Continuando desde la página ${startPage + 1} (ya tienes ${existingNeos.length} objetos)`);
}

function fetchPage(page, totalPages, allNeos, cb, retry = 0) {
  const url = `${BASE_URL}&page=${page}`;
  https.get(url, (res) => {
    let data = "";
    res.on("data", chunk => data += chunk);
    res.on("end", () => {
      let json;
      try {
        json = JSON.parse(data);
      } catch (e) {
        console.error(`Error al parsear JSON en página ${page + 1}:`, e);
        if (retry < RETRY_LIMIT) {
          console.log(`Reintentando página ${page + 1} (${retry + 1}/${RETRY_LIMIT})...`);
          return setTimeout(() => fetchPage(page, totalPages, allNeos, cb, retry + 1), PAUSE_MS);
        } else {
          return cb(allNeos);
        }
      }
      if (!json.near_earth_objects) {
        console.error(`Error en la página ${page + 1}:`, json);
        if (retry < RETRY_LIMIT) {
          console.log(`Reintentando página ${page + 1} (${retry + 1}/${RETRY_LIMIT})...`);
          return setTimeout(() => fetchPage(page, totalPages, allNeos, cb, retry + 1), PAUSE_MS);
        } else {
          return cb(allNeos);
        }
      }
      const totalObjs = json.near_earth_objects.length;
      const filteredObjs = json.near_earth_objects.filter(obj => obj.orbital_data);
      const filteredCount = totalObjs - filteredObjs.length;
      if (filteredCount > 0) {
        console.log(`Página ${page + 1}: ${filteredCount} objetos filtrados por falta de datos orbitales.`);
      }
      const neos = filteredObjs.map(obj => ({
        id: obj.id,
        name: obj.name,
        a: obj.orbital_data.semi_major_axis,
        e: obj.orbital_data.eccentricity,
        i: obj.orbital_data.inclination,
        Omega: obj.orbital_data.ascending_node_longitude,
        omega: obj.orbital_data.perihelion_argument,
        M: obj.orbital_data.mean_anomaly,
        period: obj.orbital_data.orbital_period,
        type: obj.is_potentially_hazardous_asteroid ? "PHA" : "NEO"
      }));
      allNeos.push(...neos);
      if (page < totalPages - 1) {
        setTimeout(() => fetchPage(page + 1, totalPages, allNeos, cb), PAUSE_MS);
      } else {
        cb(allNeos);
      }
    });
  }).on("error", (err) => {
    console.error(`Error de red en página ${page + 1}:`, err);
    if (retry < RETRY_LIMIT) {
      console.log(`Reintentando página ${page + 1} (${retry + 1}/${RETRY_LIMIT})...`);
      setTimeout(() => fetchPage(page, totalPages, allNeos, cb, retry + 1), PAUSE_MS);
    } else {
      cb(allNeos);
    }
  });
}

// Primero, obtenemos la cantidad total de páginas
function fetchTotalPages(retry = 0) {
  https.get(BASE_URL, (res) => {
    let data = "";
    res.on("data", chunk => data += chunk);
    res.on("end", () => {
      let json;
      try {
        json = JSON.parse(data);
      } catch (e) {
        console.error("Error al parsear JSON de la página inicial:", e);
        if (retry < RETRY_LIMIT) {
          console.log(`Reintentando obtener total de páginas (${retry + 1}/${RETRY_LIMIT})...`);
          return setTimeout(() => fetchTotalPages(retry + 1), PAUSE_MS);
        } else {
          return;
        }
      }
      if (!json.page || typeof json.page.total_pages === "undefined") {
        console.error("Error al obtener el número de páginas. Respuesta de la API:", json);
        if (retry < RETRY_LIMIT) {
          console.log(`Reintentando obtener total de páginas (${retry + 1}/${RETRY_LIMIT})...`);
          return setTimeout(() => fetchTotalPages(retry + 1), PAUSE_MS);
        } else {
          return;
        }
      }
      const totalPages = json.page.total_pages;
      console.log(`Total pages: ${totalPages}`);
      fetchPage(startPage, totalPages, existingNeos, (allNeos) => {
        fs.writeFileSync(path, JSON.stringify(allNeos, null, 2));
        console.log(`Guardados ${allNeos.length} objetos en public/data/neos.json`);
      });
    });
  }).on("error", (err) => {
    console.error("Error de red al obtener total de páginas:", err);
    if (retry < RETRY_LIMIT) {
      console.log(`Reintentando obtener total de páginas (${retry + 1}/${RETRY_LIMIT})...`);
      setTimeout(() => fetchTotalPages(retry + 1), PAUSE_MS);
    }
  });
}

// Llama a la función inicial
fetchTotalPages();