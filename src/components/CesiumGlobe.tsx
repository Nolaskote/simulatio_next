import React, { useEffect, useRef } from 'react'

type Props = {
  markerLat: number
  markerLon: number
  onPick: (latDeg: number, lonDeg: number) => void
}

export default function CesiumGlobe({ markerLat, markerLon, onPick }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const viewerRef = useRef<any>(null)
  const cesiumRef = useRef<any>(null)
  const markerId = 'impact-marker'

  useEffect(() => {
    let canceled = false
    ;(async () => {
      try {
    if (viewerRef.current) return
    // Ensure Cesium knows where to load its static assets before import side-effects
    ;(window as any).CESIUM_BASE_URL = ((import.meta as any).env?.BASE_URL || '/') + 'Cesium/'
  const Cesium = await import('cesium')
    cesiumRef.current = Cesium

        if (!containerRef.current || canceled) return
  const { Viewer, ScreenSpaceEventHandler, ScreenSpaceEventType, Cartesian3, Ellipsoid, Cartographic, Math: CMath, Color, UrlTemplateImageryProvider, EllipsoidTerrainProvider } = Cesium as any

        const viewer = new Viewer(containerRef.current, {
          animation: false,
          timeline: false,
          geocoder: false,
          homeButton: true,
          sceneModePicker: true,
          baseLayerPicker: true,
          navigationHelpButton: false,
          fullscreenButton: false,
          selectionIndicator: false,
          infoBox: false,
          // Use Esri World Imagery as the main basemap for better continent visualization
          imageryProvider: new UrlTemplateImageryProvider({
            url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            credit: '© Esri',
            minimumLevel: 0,
            maximumLevel: 19,
          }),
        })
        viewerRef.current = viewer

        // Camera behavior: center on Earth, allow rotate/zoom only (no pan/translate)
        try {
          // Initial view looking at the whole Earth
          viewer.camera.setView({
            destination: Cartesian3.fromDegrees(0, 0, 25_000_000),
          })
          const ctrl = viewer.scene.screenSpaceCameraController
          ctrl.enableTranslate = false
          ctrl.enableZoom = true
          ctrl.enableRotate = true
          // Keep tilt enabled for natural orbiting; disable free-look
          ctrl.enableLook = false
          // Constrain zoom so you can't go to infinity or inside the Earth
          ctrl.minimumZoomDistance = 700_000 // ~700 km above surface
          ctrl.maximumZoomDistance = 60_000_000 // ~60,000 km from center; Earth stays visible
          // Reduce inertia so movements stop quickly
          ctrl.inertiaSpin = 0.3
          ctrl.inertiaTranslate = 0.0
          ctrl.inertiaZoom = 0.3
        } catch {}

        // Add multiple imagery layers for better visualization
        try {
          // Add Natural Earth II imagery for better continent definition
          const naturalEarthProvider = new UrlTemplateImageryProvider({
            url: 'https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
            credit: '© Esri, National Geographic',
            minimumLevel: 0,
            maximumLevel: 16,
          })
          const naturalEarthLayer = viewer.imageryLayers.addImageryProvider(naturalEarthProvider)
          naturalEarthLayer.alpha = 0.7
          naturalEarthLayer.show = true
        } catch (e) {
          console.warn('Natural Earth imagery not available')
        }

        // Add world boundaries for clear country outlines
        try {
          const boundariesProvider = new UrlTemplateImageryProvider({
            url: 'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
            credit: 'Esri World Boundaries & Places',
            minimumLevel: 0,
            maximumLevel: 19,
          })
          const boundariesLayer = viewer.imageryLayers.addImageryProvider(boundariesProvider)
          boundariesLayer.alpha = 0.8
          boundariesLayer.show = true
        } catch (e) {
          console.warn('World Boundaries overlay not available')
        }

        // Configure terrain for better elevation visualization
        try {
          viewer.terrainProvider = new EllipsoidTerrainProvider()
          // Enable terrain exaggeration for better visual effect
          viewer.scene.globe.terrainExaggeration = 1.0
          // Set globe material properties for better appearance
          viewer.scene.globe.enableLighting = true
          viewer.scene.globe.dynamicAtmosphereLighting = true
          viewer.scene.globe.atmosphereIntensity = 0.1
        } catch (e) {
          console.warn('Terrain configuration failed')
        }

        // Initial marker
        viewer.entities.add({
          id: markerId,
          position: Cartesian3.fromDegrees(markerLon, markerLat),
          point: { pixelSize: 10, color: Color.RED, outlineColor: Color.WHITE, outlineWidth: 2 }
        })
        // Do not auto-fly to the marker; keep the centered world view

        // Click handler to set marker and report lat/lon
        const handler = new ScreenSpaceEventHandler(viewer.scene.canvas)
        handler.setInputAction((movement: any) => {
          if (!viewer || viewer.isDestroyed && viewer.isDestroyed()) return
          const cartesian = viewer.camera.pickEllipsoid(movement.position, Ellipsoid.WGS84)
          if (!cartesian) return
          const carto = Cartographic.fromCartesian(cartesian)
          const latDeg = CMath.toDegrees(carto.latitude)
          const lonDeg = CMath.toDegrees(carto.longitude)
          const ent = viewer.entities.getById(markerId)
          if (ent) ent.position = Cartesian3.fromDegrees(lonDeg, latDeg)
          onPick(latDeg, lonDeg)
        }, ScreenSpaceEventType.LEFT_CLICK)
        // Save handler for cleanup
        ;(viewer as any).__pickHandler = handler

      } catch (e) {
        console.warn('Cesium init failed', e)
        try {
          const v = viewerRef.current
          if (v && !v.isDestroyed()) v.destroy()
        } catch {}
      }
    })()
    return () => {
      canceled = true
      try {
        const viewer = viewerRef.current
        if (viewer) {
          try {
            const h = (viewer as any).__pickHandler
            if (h) h.destroy()
          } catch {}
          viewer.destroy()
        }
      } catch {}
      viewerRef.current = null
    }
  }, [])

  // Sync marker when props change
  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer) return
    try {
      const Cesium = cesiumRef.current
      const Cartesian3 = Cesium?.Cartesian3
      const ent = viewer.entities.getById(markerId)
      if (ent && Cartesian3) ent.position = Cartesian3.fromDegrees(markerLon, markerLat)
    } catch {}
  }, [markerLat, markerLon])

  return <div ref={containerRef} style={{ position: 'absolute', inset: 0 }} />
}
