declare module 'react-simple-maps' {
  import * as React from 'react';
  
  export interface ComposableMapProps {
    projection?: string;
    projectionConfig?: any;
    width?: number;
    height?: number;
    style?: React.CSSProperties;
    children?: React.ReactNode;
  }
  
  export interface GeographiesProps {
    geography: string | any;
    children: (props: { geographies: any[] }) => React.ReactNode;
  }
  
  export interface GeographyProps {
    geography: any;
    style?: {
      default?: React.CSSProperties;
      hover?: React.CSSProperties;
      pressed?: React.CSSProperties;
    };
    onMouseEnter?: (event: React.MouseEvent) => void;
    onMouseLeave?: (event: React.MouseEvent) => void;
    onClick?: (event: React.MouseEvent) => void;
  }
  
  export interface ZoomableGroupProps {
    center?: [number, number];
    zoom?: number;
    onMoveStart?: (coordinates: any, zoom: number) => void;
    onMoveEnd?: (coordinates: any) => void;
    onZoomStart?: (coordinates: any, zoom: number) => void;
    onZoomEnd?: (coordinates: any, zoom: number) => void;
    disablePanning?: boolean;
    disableZooming?: boolean;
    children?: React.ReactNode;
  }
  
  export interface MarkerProps {
    coordinates: [number, number];
    children?: React.ReactNode;
    onClick?: (event: React.MouseEvent) => void;
    onMouseEnter?: (event: React.MouseEvent) => void;
    onMouseLeave?: (event: React.MouseEvent) => void;
    style?: {
      default?: React.CSSProperties;
      hover?: React.CSSProperties;
      pressed?: React.CSSProperties;
    };
  }
  
  export const ComposableMap: React.FC<ComposableMapProps>;
  export const Geographies: React.FC<GeographiesProps>;
  export const Geography: React.FC<GeographyProps>;
  export const ZoomableGroup: React.FC<ZoomableGroupProps>;
  export const Marker: React.FC<MarkerProps>;
}