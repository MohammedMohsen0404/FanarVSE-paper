/* Batched asynchronous random reads of sector-aligned rows (one row = one 4096-byte sector) with overlapped I/O and an I/O completion port.
   read_rows(handle, port, ids, n, out, row_bytes, qd, bufs): reads row ids[i] into out + i*row_bytes, keeping up to qd requests in flight. */
#include <windows.h>
#define SECTOR 4096
typedef struct { ULONG_PTR lpCompletionKey; LPOVERLAPPED lpOverlapped; ULONG_PTR Internal; DWORD dwNumberOfBytesTransferred; } OVENT;
typedef BOOL (WINAPI *GQCSEx_t)(HANDLE, OVENT*, ULONG, PULONG, DWORD, BOOL);
static GQCSEx_t gqcs = NULL;

__declspec(dllexport) void* open_tier(const wchar_t* path) {
    return CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, NULL);
}
__declspec(dllexport) void* make_port(void* h) { return CreateIoCompletionPort(h, NULL, 0, 1); }
__declspec(dllexport) void* alloc_bufs(int qd) { return VirtualAlloc(NULL, (SIZE_T)qd * SECTOR, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE); }

static int issue(HANDLE h, OVERLAPPED* ov, char* buf, long long row) {
    long long off = row * (long long)SECTOR;
    ZeroMemory(ov, sizeof(OVERLAPPED)); ov->Offset = (DWORD)(off & 0xFFFFFFFF); ov->OffsetHigh = (DWORD)(off >> 32);
    if (!ReadFile(h, buf, SECTOR, NULL, ov) && GetLastError() != ERROR_IO_PENDING) return -1;
    return 0;
}

__declspec(dllexport) int read_rows(void* hv, void* portv, const long long* ids, int n, char* out, int row_bytes, int qd, char* bufs) {
    HANDLE h = (HANDLE)hv, port = (HANDLE)portv;
    OVERLAPPED ov[256]; int slot_item[256]; int next = 0, inflight = 0, done = 0, s;
    if (qd > 256) qd = 256;
    for (s = 0; s < qd && next < n; s++) { slot_item[s] = next; if (issue(h, &ov[s], bufs + (size_t)s * SECTOR, ids[next])) return -1; next++; inflight++; }
    while (done < n) {
        OVENT ent[64]; ULONG got = 0; int k;
        if (!gqcs) gqcs = (GQCSEx_t)GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetQueuedCompletionStatusEx");
        if (!gqcs(port, ent, 64, &got, INFINITE, FALSE)) return -2;
        for (k = 0; k < (int)got; k++) {
            int sl = (int)(ent[k].lpOverlapped - ov); int it = slot_item[sl];
            memcpy(out + (size_t)it * row_bytes, bufs + (size_t)sl * SECTOR, row_bytes); done++; inflight--;
            if (next < n) { slot_item[sl] = next; if (issue(h, &ov[sl], bufs + (size_t)sl * SECTOR, ids[next])) return -1; next++; inflight++; }
        }
    }
    return done;
}
