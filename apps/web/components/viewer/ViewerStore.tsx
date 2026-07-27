"use client";

import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import type { Vector3 } from "@/lib/atlas-schema/types";
import type { CoverageMetric } from "@/lib/data/metrics";
import type { VoxelSelection } from "@/lib/data/radiusQuery";

interface ViewerState {
  leftVisible: boolean;
  rightVisible: boolean;
  cameraResetToken: number;
  metric: CoverageMetric;
  spacing: number;
  radius: number;
  selection: VoxelSelection | null;
  autoRotate: boolean;
  toggleArm: (arm: "left" | "right") => void;
  resetCamera: () => void;
  setMetric: (metric: CoverageMetric) => void;
  setSpacing: (spacing: number) => void;
  setRadius: (radius: number) => void;
  selectVoxel: (
    arm: "left" | "right",
    voxelEntryIndex: number,
    exportedCenter: Vector3,
  ) => void;
  clearSelection: () => void;
  setAutoRotate: (enabled: boolean) => void;
}

const ViewerStoreContext = createContext<ViewerState | null>(null);

export function ViewerStore({ children }: { children: ReactNode }) {
  const [leftVisible, setLeftVisible] = useState(true);
  const [rightVisible, setRightVisible] = useState(true);
  const [cameraResetToken, setCameraResetToken] = useState(0);
  const [metric, setMetric] = useState<CoverageMetric>("visits");
  const [spacing, setSpacingState] = useState(0.8);
  const [radius, setRadius] = useState(0.05);
  const [selection, setSelection] = useState<VoxelSelection | null>(null);
  const [autoRotate, setAutoRotate] = useState(false);
  const toggleArm = useCallback((arm: "left" | "right") => {
    if (arm === "left") setLeftVisible((value) => !value);
    else setRightVisible((value) => !value);
  }, []);
  const resetCamera = useCallback(
    () => setCameraResetToken((value) => value + 1),
    [],
  );
  const setSpacing = useCallback((value: number) => {
    setSpacingState(value);
    setSelection(null);
  }, []);
  const selectVoxel = useCallback(
    (
      arm: "left" | "right",
      voxelEntryIndex: number,
      exportedCenter: Vector3,
    ) => setSelection({ arm, voxelEntryIndex, exportedCenter }),
    [],
  );
  const clearSelection = useCallback(() => setSelection(null), []);
  const value = useMemo(
    () => ({
      leftVisible,
      rightVisible,
      cameraResetToken,
      metric,
      spacing,
      radius,
      selection,
      autoRotate,
      toggleArm,
      resetCamera,
      setMetric,
      setSpacing,
      setRadius,
      selectVoxel,
      clearSelection,
      setAutoRotate,
    }),
    [
      leftVisible,
      rightVisible,
      cameraResetToken,
      metric,
      spacing,
      radius,
      selection,
      autoRotate,
      toggleArm,
      resetCamera,
      setSpacing,
      selectVoxel,
      clearSelection,
    ],
  );
  return (
    <ViewerStoreContext.Provider value={value}>
      {children}
    </ViewerStoreContext.Provider>
  );
}

export function useViewerStore() {
  const value = useContext(ViewerStoreContext);
  if (!value) throw new Error("useViewerStore must be used inside ViewerStore.");
  return value;
}
