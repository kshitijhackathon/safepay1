import { QueryClient, QueryFunction } from "@tanstack/react-query";

async function throwIfResNotOk(res: Response) {
  if (!res.ok) {
    let errorMessage;
    try {
      // Try to parse as JSON first
      const errorData = await res.json();
      errorMessage = errorData.message || errorData.error || res.statusText;
    } catch (e) {
      // Fallback to plain text if not JSON
      const text = await res.text();
      errorMessage = text || res.statusText;
    }
    throw new Error(`${res.status}: ${errorMessage}`);
  }
}

export async function apiRequest(
  method: string,
  url: string,
  data?: unknown | undefined,
  isFormData: boolean = false,
): Promise<Response> {
  try {
    let fetchOptions: RequestInit = {
      method,
      credentials: "include",
    };

    if (data) {
      if (isFormData) {
        // Don't set Content-Type for FormData - browser will set it with the boundary
        fetchOptions.body = data as FormData;
      } else {
        fetchOptions.headers = { "Content-Type": "application/json" };
        fetchOptions.body = JSON.stringify(data);
      }
    }

    const res = await fetch(url, fetchOptions);

    // Clone the response before checking it to avoid body stream issues
    const resClone = res.clone();
    await throwIfResNotOk(resClone);
    return res;
  } catch (error) {
    console.error(`API Request error for ${method} ${url}:`, error);
    throw error;
  }
}

type UnauthorizedBehavior = "returnNull" | "throw";
export const getQueryFn: <T>(options: {
  on401: UnauthorizedBehavior;
}) => QueryFunction<T> =
  ({ on401: unauthorizedBehavior }) =>
  async ({ queryKey }) => {
    const res = await fetch(queryKey[0] as string, {
      credentials: "include",
    });

    if (unauthorizedBehavior === "returnNull" && res.status === 401) {
      return null;
    }

    // Clone the response before checking it to avoid body stream issues
    const resClone = res.clone();
    await throwIfResNotOk(resClone);
    return await res.json();
  };

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: getQueryFn({ on401: "throw" }),
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
      retry: false,
    },
    mutations: {
      retry: false,
    },
  },
});
