"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { GeoJsonObject } from "geojson";
import type { GeoJSONData, MapLabels } from "@/types";

const COLORS = [
  "#2563eb",
  "#dc2626",
  "#16a34a",
  "#d97706",
  "#7c3aed",
  "#0891b2",
  "#be185d",
  "#65a30d",
  "#9333ea",
  "#0f766e",
];
const NOISE_COLOR = "#9ca3af";

const DEFAULT_UI = {
  name: "Name",
  type: "Type",
  address: "Address",
  cluster: "Cluster",
  unclustered: "Unclustered",
  legendTitle: "Legend",
  noName: "(no name)",
};

function buildAddress(props: Record<string, unknown>): string {
  const get = (k: string) => {
    const v = props[k];
    return typeof v === "string" && v.trim() ? v.trim() : "";
  };
  const street = get("addr:street");
  const house = get("addr:housenumber");
  const city = get("addr:city");
  const postcode = get("addr:postcode");
  const state = get("addr:state");
  const country = get("addr:country");

  const line1 = [street, house].filter(Boolean).join(" ");
  const line2 = [postcode, city].filter(Boolean).join(" ");
  return [line1, line2, state, country].filter(Boolean).join(", ");
}

const ICON_MAX = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>`;
const ICON_MIN = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 14 10 14 10 20"></polyline><polyline points="20 10 14 10 14 4"></polyline><line x1="14" y1="10" x2="21" y2="3"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>`;

function clusterColor(clust: number | undefined | null) {
  if (clust === undefined || clust === null || clust < 0) return NOISE_COLOR;
  return COLORS[clust % COLORS.length];
}

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function FitBounds({ geojson, trigger }: { geojson: GeoJSONData; trigger: unknown }) {
  const map = useMap();
  useEffect(() => {
    try {
      const layer = L.geoJSON(geojson as unknown as GeoJsonObject);
      const bounds = layer.getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [24, 24], maxZoom: 15 });
      }
    } catch {
      // ignore invalid bounds
    }
  }, [geojson, map, trigger]);
  return null;
}

function InvalidateOnChange({ trigger }: { trigger: unknown }) {
  const map = useMap();
  useEffect(() => {
    const t = setTimeout(() => map.invalidateSize(), 60);
    return () => clearTimeout(t);
  }, [trigger, map]);
  return null;
}

function ZoomMode({ fullscreen }: { fullscreen: boolean }) {
  const map = useMap();
  useEffect(() => {
    if (fullscreen) map.scrollWheelZoom.enable();
    else map.scrollWheelZoom.disable();
  }, [fullscreen, map]);
  return null;
}

function FullscreenControl({
  fullscreen,
  onToggle,
  exitTitle,
  enterTitle,
}: {
  fullscreen: boolean;
  onToggle: () => void;
  exitTitle: string;
  enterTitle: string;
}) {
  const map = useMap();
  const onToggleRef = useRef(onToggle);
  onToggleRef.current = onToggle;

  useEffect(() => {
    const zoomCtl = (map as L.Map & { zoomControl?: L.Control.Zoom }).zoomControl;
    const container = zoomCtl?.getContainer();
    if (!container) return;

    const btn = L.DomUtil.create(
      "a",
      "leaflet-control-zoom-fullscreen",
      container
    ) as HTMLAnchorElement;
    btn.href = "#";
    btn.setAttribute("role", "button");
    const title = fullscreen ? exitTitle : enterTitle;
    btn.title = title;
    btn.setAttribute("aria-label", title);
    btn.style.display = "flex";
    btn.style.alignItems = "center";
    btn.style.justifyContent = "center";
    btn.innerHTML = fullscreen ? ICON_MIN : ICON_MAX;

    const handler = (e: Event) => {
      L.DomEvent.preventDefault(e);
      L.DomEvent.stopPropagation(e);
      onToggleRef.current();
    };
    L.DomEvent.on(btn, "click", handler);

    return () => {
      L.DomEvent.off(btn, "click", handler);
      btn.remove();
    };
  }, [map, fullscreen, exitTitle, enterTitle]);

  return null;
}

export default function MapView({
  geojson,
  labels,
}: {
  geojson: GeoJSONData;
  labels?: MapLabels;
}) {
  const ui = { ...DEFAULT_UI, ...(labels?.ui ?? {}) };
  const clusterLabels = labels?.clusters ?? {};
  const [legendOpen, setLegendOpen] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFullscreen(false);
    };
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prevOverflow;
      window.removeEventListener("keydown", onKey);
    };
  }, [fullscreen]);

  const presentClusters = useMemo(() => {
    const set = new Set<number>();
    for (const f of geojson.features) {
      const c = (f.properties?.clust as number | undefined) ?? null;
      if (typeof c === "number") set.add(c);
    }
    return Array.from(set).sort((a, b) => a - b);
  }, [geojson]);

  const labelForCluster = (clust: number) => {
    const explicit = clusterLabels[String(clust)];
    if (explicit) return explicit;
    if (clust < 0) return ui.unclustered;
    return `${ui.cluster} ${clust}`;
  };

  const buildPopup = (props: Record<string, unknown>) => {
    const name = (props.name as string) || `<em>${escapeHtml(ui.noName)}</em>`;
    const feature = (props.feature as string) || "";
    const clust = props.clust as number | undefined;
    const clusterLabel =
      typeof clust === "number" ? labelForCluster(clust) : "";
    const color = clusterColor(clust);

    const rows: string[] = [];
    rows.push(
      `<div style="font-weight:600;color:#18181b;margin-bottom:4px;">${
        name.startsWith("<em>") ? name : escapeHtml(name)
      }</div>`
    );
    if (feature) {
      rows.push(
        `<div style="font-size:11px;color:#52525b;"><span style="color:#71717a;">${escapeHtml(
          ui.type
        )}:</span> ${escapeHtml(feature)}</div>`
      );
    }
    const address = buildAddress(props);
    if (address) {
      rows.push(
        `<div style="font-size:11px;color:#52525b;margin-top:2px;"><span style="color:#71717a;">${escapeHtml(
          ui.address
        )}:</span> ${escapeHtml(address)}</div>`
      );
    }
    if (clusterLabel) {
      rows.push(
        `<div style="font-size:11px;color:#52525b;display:flex;align-items:center;gap:6px;margin-top:2px;">
           <span style="display:inline-block;width:10px;height:10px;border-radius:9999px;background:${color};"></span>
           <span>${escapeHtml(clusterLabel)}</span>
         </div>`
      );
    }
    return `<div style="min-width:140px;font-family:inherit;">${rows.join("")}</div>`;
  };

  const containerClass = fullscreen
    ? "fixed inset-0 z-[9999] bg-white"
    : "w-full h-72 rounded-xl overflow-hidden border border-zinc-100 mt-3 isolate relative z-0";

  return (
    <div className={containerClass} style={{ touchAction: "none" }}>
      <MapContainer
        center={[0, 0]}
        zoom={2}
        style={{ width: "100%", height: "100%" }}
        scrollWheelZoom={false}
        touchZoom={true}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
          maxZoom={20}
          subdomains="abcd"
        />
        <GeoJSON
          data={geojson as unknown as GeoJsonObject}
          pointToLayer={(feature, latlng) => {
            const color = clusterColor(feature.properties?.clust as number);
            return L.circleMarker(latlng, {
              radius: 6,
              fillColor: color,
              color: "#ffffff",
              weight: 1.5,
              opacity: 1,
              fillOpacity: 0.85,
            });
          }}
          style={(feature) => {
            const color = clusterColor(feature?.properties?.clust as number);
            return { color, weight: 2, opacity: 0.8, fillOpacity: 0.35 };
          }}
          onEachFeature={(feature, layer) => {
            layer.bindPopup(buildPopup(feature.properties ?? {}));
            const name = feature.properties?.name as string | undefined;
            const featureType = feature.properties?.feature as string | undefined;
            const tooltip = name || featureType;
            if (tooltip) {
              layer.bindTooltip(escapeHtml(tooltip), {
                direction: "top",
                sticky: true,
                opacity: 0.9,
              });
            }
          }}
        />
        <FitBounds geojson={geojson} trigger={fullscreen} />
        <InvalidateOnChange trigger={fullscreen} />
        <ZoomMode fullscreen={fullscreen} />
        <FullscreenControl
          fullscreen={fullscreen}
          onToggle={() => setFullscreen((f) => !f)}
          enterTitle="Fullscreen"
          exitTitle="Exit fullscreen (Esc)"
        />
      </MapContainer>

      {/* Legend */}
      {presentClusters.length > 0 && (
        <div
          className="absolute top-2 right-2 z-[400] bg-white/95 backdrop-blur rounded-lg shadow-md border border-zinc-200 text-[11px] text-zinc-700 overflow-hidden"
          style={{ width: 115 }}
        >
          <button
            onClick={() => setLegendOpen((o) => !o)}
            className="flex items-center justify-between w-full gap-2 px-2.5 py-1 hover:bg-zinc-50 transition-colors font-medium text-zinc-800"
            aria-expanded={legendOpen}
          >
            <span className="truncate">{ui.legendTitle}</span>
            {legendOpen ? (
              <ChevronUp size={12} className="text-zinc-500 flex-shrink-0" />
            ) : (
              <ChevronDown size={12} className="text-zinc-500 flex-shrink-0" />
            )}
          </button>
          {legendOpen && (
            <ul
              className="px-2.5 pb-1.5 pt-0.5 space-y-1 overflow-y-auto"
              style={{ maxHeight: 120 }}
            >
              {presentClusters.map((c) => (
                <li key={c} className="flex items-start gap-2">
                  <span
                    className="inline-block w-2.5 h-2.5 rounded-full border border-white shadow-sm flex-shrink-0"
                    style={{ background: clusterColor(c), marginTop: 3 }}
                  />
                  <span className="leading-tight line-clamp-2 break-words">
                    {labelForCluster(c)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
