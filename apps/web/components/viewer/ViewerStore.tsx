"use client";

import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";

interface ViewerState {
  leftVisible: boolean;
  rightVisible: boolean;
  cameraResetToken: number;
  toggleArm: (arm: "left" | "right") => void;
  resetCamera: () => void;
}

const ViewerStoreContext = createContext<ViewerState | null>(null);

export function ViewerStore({ children }: { children: ReactNode }) {
  const [leftVisible, setLeftVisible] = useState(true);
  const [rightVisible, setRightVisible] = useState(true);
  const [cameraResetToken, setCameraResetToken] = useState(0);
  const toggleArm = useCallback((arm: "left" | "right") => {
    if (arm === "left") setLeftVisible((value) => !value);
    else setRightVisible((value) => !value);
  }, []);
  const resetCamera = useCallback(
    () => setCameraResetToken((value) => value + 1),
    [],
  );
  const value = useMemo(
    () => ({
      leftVisible,
      rightVisible,
      cameraResetToken,
      toggleArm,
      resetCamera,
    }),
    [
      leftVisible,
      rightVisible,
      cameraResetToken,
      toggleArm,
      resetCamera,
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
