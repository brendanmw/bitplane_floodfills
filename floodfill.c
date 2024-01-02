/*******************************************************************************************
*
* Bitplane floodfill demos rendered with Raylib.
*
********************************************************************************************/

#include "raylib.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <intrin.h>

#include "profileapi.h"

#define DARKDARKBLUE   CLITERAL(Color){ 0, 71, 141, 255 } 

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32; 
typedef unsigned long long uint64;
  

uint64 CPUFreq = 0;
uint64 OSFreq = 0;

static inline uint64 ReadTSC()
{
    return __rdtsc();
}

static inline double CyclesToSeconds(uint64 cpuCycles)
{
    return (double)cpuCycles / (double)CPUFreq;
}

void InitializeTSCFrequency()
{
    LARGE_INTEGER counterFreq;
    QueryPerformanceFrequency(&counterFreq);
    
    OSFreq = counterFreq.QuadPart;
    
    LARGE_INTEGER startCount, curCount;
    QueryPerformanceCounter(&startCount);
    curCount = startCount;
 
    uint64 tscStart = ReadTSC();
 
    // Spin for 50 ms. We know this will be 50ms because QPC frequency is always 10MHz.
    uint64 calibrationInterval = OSFreq / 20;
    while (curCount.QuadPart - startCount.QuadPart < calibrationInterval)
    {
        QueryPerformanceCounter(&curCount);
    }
    
    uint64 osInterval = curCount.QuadPart - startCount.QuadPart;
    uint64 tscInterval = ReadTSC() - tscStart;
    CPUFreq = (tscInterval*OSFreq) / osInterval;
    
    printf("Detected timer frequencies: OS: %lld  CPU: %lld\n", OSFreq, CPUFreq);
}


  
typedef struct 
{
    int RowIndex;
    int Stage;
    uint64 FillRowPrev;
    uint64 Test;
    bool PushNextAbove;
    bool PushNextBelow;
} SimulSpanFillSIState;

typedef struct
{
    int CellIndex;
    int Stage;
    uint8 PushTop : 1;
    uint8 PushBottom : 1;
    uint8 PushLeft : 1;
    uint8 PushRight : 1;
} DFSSIState;

typedef struct
{
    int CellIndex;
    int Stage;
    int X;
    int xRight;
    int xLeft;
    int PrevSeed;
    int PushCount;
} SpanFillSIState;

typedef union
{
    DFSSIState Dfs;
    SpanFillSIState Sf;
    SimulSpanFillSIState Ssf;
} IncrementalState;
  
// Simultaneous span fill algorithm is specific to 64x64 plane. 
// Planes of 128x128 or 256x256 could be handled in the same fashion with SSE and AVX respectively.
// Larger planes can be handled by segmentation with, extra cases for horizontal span edges.
// Planes of non-aligned sizes can be further handled by introducing intermediate copying for the working rows.
// All of this would add general cost and complexity.

// The algorithm can extend to 3D fills trivially by applying 4-way DFS in two dimensions, but we can do even
// better by fitting entire decks of bits in SSE registers and doing fill operations on whole decks at once.

const int dim = 64;
const size_t decksize = (dim*dim)/8;
const int numAlgos = 3;

// Switched on algo
int Flood(int algo, const uint8* bitdeck, int dim, uint8* filled, int seedX, int seedY);
int Flood_Incremental(int algo, const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, uint8* tested, int* numTested);
int Flood_Incremental_Start(int algo, const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, int seedX, int seedY);

// DFS with stack
int Flood_1(const uint8* bitdeck, int dim, uint8* filled, int seedX, int seedY);
int Flood_1_Incremental(const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, uint8* tested, int* numTested);
int Flood_1_Incremental_Start(const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, int seedX, int seedY);

// Span fill
int Flood_2(const uint8* bitdeck, int dim, uint8* filled, int seedX, int seedY);
int Flood_2_Incremental(const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, uint8* tested, int* numTested);
int Flood_2_Incremental_Start(const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, int seedX, int seedY);

// Simultaneous Span fill 64 bit
int Flood_3(const uint8* bitdeck, int dim, uint8* filled, int seedX, int seedY);
int Flood_3_Incremental(const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, uint8* tested, int* numTested);
int Flood_3_Incremental_Start(const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, int seedX, int seedY);

static int* SFI_Stack = 0;
static int SFI_StackCount = 0;

void SFI_StackInit(int dim)
{
    SFI_Stack = malloc(dim*sizeof(int));
}

void SFI_StackFree()
{
    free(SFI_Stack);
}

void SFI_StackPush(int index)
{
    SFI_Stack[SFI_StackCount++] = index;
}

int SFI_StackPop()
{
    return SFI_Stack[--SFI_StackCount];
}

int SFI_StackSize()
{
    return SFI_StackCount;
}

uint64 CountBits(uint64 val)
{
    return __popcnt64(val);
}

size_t Max(size_t a, size_t b)
{
    return a >= b ? a : b;
}

const char* AlgoName(int algoIndex)
{
    switch(algoIndex)
    {
        case 0: return "Four-Way DFS";
        case 1: return "Span Fill";
        case 2: return "Simul Span Fill";
        default: return "";
    }
}

void ResetDeck(uint8* deck)
{
    memset(deck, 0, decksize);
}

void FillDeck(uint8* deck)
{
    memset(deck, 0xff, decksize);
}

void SaveDeck(uint8* deck, const char* file)
{
    FILE* fh = fopen(file, "wb");
    if (!fh) return;
    
    fwrite(deck, decksize, 1, fh);
    fclose(fh);
}

void LoadDeck(uint8* deck, const char* file)
{
    FILE* fh = fopen(file, "rb");
    if (!fh) return;
    
    fread(deck, decksize, 1, fh);
    fclose(fh);
}

void FillWorstCase(uint8* deck)
{
     // This is an approximate worst case for scan fill
    uint64* ull = (uint64*)deck;
    // b01010101 
    // b01110111 77
    // b01010101
    // b11011101 dd
    ull[0] = 0x5555555555555555llu;
    ull[1] = 0x7777777777777777llu;
    ull[2] = 0x5555555555555555llu;
    ull[3] = 0xddddddddddddddddllu;
    ull[4] = 0x5555555555555555llu;
    ull[5] = 0x7777777777777777llu;
    ull[6] = 0x5555555555555555llu;
    ull[7] = 0xddddddddddddddddllu;
    ull[8] = 0x5555555555555555llu;
    ull[9] = 0x7777777777777777llu;
    ull[10] = 0x5555555555555555llu;
    ull[11] = 0xddddddddddddddddllu;
    ull[12] = 0x5555555555555555llu;
    ull[13] = 0x7777777777777777llu;
    ull[14] = 0x5555555555555555llu;
    ull[15] = 0xddddddddddddddddllu;
    ull[16] = 0x5555555555555555llu;
    ull[17] = 0x7777777777777777llu;
    ull[18] = 0x5555555555555555llu;
    ull[19] = 0xddddddddddddddddllu;
    ull[20] = 0x5555555555555555llu;
    ull[21] = 0x7777777777777777llu;
    ull[22] = 0x5555555555555555llu;
    ull[23] = 0xddddddddddddddddllu;
    ull[24] = 0x5555555555555555llu;
    ull[25] = 0x7777777777777777llu;
    ull[26] = 0x5555555555555555llu;
    ull[27] = 0xddddddddddddddddllu;
    ull[28] = 0x5555555555555555llu;
    ull[29] = 0x7777777777777777llu;
    ull[30] = 0x5555555555555555llu;
    ull[31] = 0xddddddddddddddddllu;
    ull[32] = 0x5555555555555555llu;
    ull[33] = 0x7777777777777777llu;
    ull[34] = 0x5555555555555555llu;
    ull[35] = 0xddddddddddddddddllu;
    ull[36] = 0x5555555555555555llu;
    ull[37] = 0x7777777777777777llu;
    ull[38] = 0x5555555555555555llu;
    ull[39] = 0xddddddddddddddddllu;
    ull[40] = 0x5555555555555555llu;
    ull[41] = 0x7777777777777777llu;
    ull[42] = 0x5555555555555555llu;
    ull[43] = 0xddddddddddddddddllu;
    ull[44] = 0x5555555555555555llu;
    ull[45] = 0x7777777777777777llu;
    ull[46] = 0x5555555555555555llu;
    ull[47] = 0xddddddddddddddddllu;
    ull[48] = 0x5555555555555555llu;
    ull[49] = 0x7777777777777777llu;
    ull[50] = 0x5555555555555555llu;
    ull[51] = 0xddddddddddddddddllu;
    ull[52] = 0x5555555555555555llu;
    ull[53] = 0x7777777777777777llu;
    ull[54] = 0x5555555555555555llu;
    ull[55] = 0xddddddddddddddddllu;
    ull[56] = 0x5555555555555555llu;
    ull[57] = 0x7777777777777777llu;
    ull[58] = 0x5555555555555555llu;
    ull[59] = 0xddddddddddddddddllu;
    ull[60] = 0x5555555555555555llu;
    ull[61] = 0x7777777777777777llu;
    ull[62] = 0x5555555555555555llu;
    ull[63] = 0xddddddddddddddddllu;
}

//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------
int main(void)
{
    // Initialization
    //--------------------------------------------------------------------------------------
    const int rectSize = 15;
    const int rectMargin = 1;
    const int rectSpacing = rectSize+rectMargin;
    const int topMargin = 30;
    
    const int screenWidth = rectSpacing*dim + rectMargin;
    const int screenHeight = rectSpacing*dim + rectMargin + topMargin;
    
    InitializeTSCFrequency();

    InitWindow(screenWidth, screenHeight, "Bitplane Floodfill Tests");

    SetTargetFPS(60);               // Set our game to run at 60 frames-per-second
    //--------------------------------------------------------------------------------------



    // 16x16 deck of bits

    uint8* bitdeck = malloc(decksize);
    uint8* filled = malloc(decksize);
    uint8* visited = malloc(decksize);
    uint8* tested = malloc(decksize);
    
    // Initial config 
    FillDeck(bitdeck);
    ResetDeck(filled);
    ResetDeck(visited);
    ResetDeck(tested);
    
    
    unsigned int algoIndex = 0;
    bool stepMode = false;
    int iterationsPerFrame = 1;
    
    char textBuf[1024];
    int lastFilledCount = 0;
    
    // Just as big as possibly required.
    int* incrementalFillStack = malloc(sizeof(IncrementalState)*dim*dim);
    int incrementalFillStackCount = 0;
    
    SFI_StackInit(dim);
    
    int maxStackSize = 0;
    int totalTested = 0;

    double lastRuntimeUS = 0.0f;

    // Main game loop
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        // Update
        
        Vector2 mouseScreenPos = GetMousePosition();
        int cellX = ((int)mouseScreenPos.x / rectSpacing);
        int cellY = ((int)mouseScreenPos.y-topMargin) / rectSpacing;
            
        if (IsKeyPressed(KEY_S))
        {
            stepMode = !stepMode;
        }            
        
        if (IsKeyPressed(KEY_KP_ADD))
        {
            iterationsPerFrame++;
        }
        if (IsKeyPressed(KEY_KP_SUBTRACT))
        {
            iterationsPerFrame = Max(1, iterationsPerFrame-1);
        }
            
        if (incrementalFillStackCount > 0)
        {
            if (IsKeyPressed(KEY_ENTER))
            {
                ResetDeck(tested);
                
                while (incrementalFillStackCount)
                {
                    int testCount = 0;
                    lastFilledCount +=  Flood_Incremental(algoIndex, bitdeck, dim, filled, incrementalFillStack, &incrementalFillStackCount, tested, &testCount);
                    
                    maxStackSize = incrementalFillStackCount > maxStackSize ? incrementalFillStackCount : maxStackSize;
                    totalTested += testCount;
                }
                
                ResetDeck(tested);
            }
            else if (!stepMode || IsKeyPressed(KEY_SPACE) || IsKeyPressedRepeat(KEY_SPACE))
            {
                ResetDeck(tested);
                      
                int remainingIterations = iterationsPerFrame;
                while (remainingIterations && incrementalFillStackCount)
                {
                    int testCount = 0;
                    lastFilledCount +=  Flood_Incremental(algoIndex, bitdeck, dim, filled, incrementalFillStack, &incrementalFillStackCount, tested, &testCount);
                    
                    maxStackSize = incrementalFillStackCount > maxStackSize ? incrementalFillStackCount : maxStackSize;
                    totalTested += testCount;
                    
                    --remainingIterations;
                }
            }
        }
        else
        {
            ResetDeck(tested);
            
            if (IsKeyPressed(KEY_SPACE) || IsMouseButtonPressed(MOUSE_BUTTON_RIGHT))
            {
                ResetDeck(filled);
            }
            
            if (IsKeyPressed(KEY_GRAVE))
            {
                FillDeck(bitdeck);
            }
            
            if (IsKeyPressed(KEY_W))
            {
                FillWorstCase(bitdeck);
            }
            
            if (IsKeyPressed(KEY_L))
            {
                LoadDeck(bitdeck, "saved.bitplane");
            }
            
            if (IsKeyPressed(KEY_F))
            {
                SaveDeck(bitdeck, "saved.bitplane");
            }
            
            if (IsKeyPressed(KEY_DOWN))
            {
                algoIndex = (algoIndex + 1) % numAlgos;
            }
            else if (IsKeyPressed(KEY_UP))
            {
                algoIndex = (algoIndex + numAlgos - 1) % numAlgos;
            }
            
            if (IsMouseButtonDown(MOUSE_BUTTON_LEFT))
            {
                int bitIndex = cellY*dim + cellX;
                int byte = bitIndex/8;
                int bit = bitIndex%8;
                if (IsKeyDown(KEY_LEFT_SHIFT))
                {
                   bitdeck[byte] = bitdeck[byte] | (1 << bit);
                    
                }
                else
                {
                   bitdeck[byte] = bitdeck[byte] & ~(1 << bit);
                }
            }
            
            if (IsMouseButtonPressed(MOUSE_BUTTON_MIDDLE))
            {
                maxStackSize = 0;
                totalTested = 0;
                
                 if (IsKeyDown(KEY_LEFT_SHIFT))
                 {
                    lastFilledCount = Flood_Incremental_Start(algoIndex, bitdeck, dim, filled, incrementalFillStack, &incrementalFillStackCount, cellX, cellY);
                 }
                 else
                 {
                    uint64 startCycles = ReadTSC();
                     
                    lastFilledCount = Flood(algoIndex, bitdeck, dim, filled, cellX, cellY);
                    
                    uint64 interval = ReadTSC() - startCycles;
                    lastRuntimeUS = CyclesToSeconds(interval)*1000000;
                 }
            }
        }
        


        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();

            ClearBackground(RAYWHITE);
            
            sprintf(textBuf, "Screen pos: (%04d,%04d)  Last filled count: %d  Stack size: %d  Max stack size: %d Total reads: %d Last time: %.3fus", 
                (int)mouseScreenPos.x, 
                (int)mouseScreenPos.y, 
                lastFilledCount, 
                incrementalFillStackCount, 
                maxStackSize, 
                totalTested,
                lastRuntimeUS);
                
            DrawText(textBuf, 0, 0, 12, BLACK);
            
            sprintf(textBuf, "Algo name: %s  Step mode: %s  Speed: %d", AlgoName(algoIndex), stepMode ? "on" : "off", iterationsPerFrame);
            
            DrawText(textBuf, 0, 14, 12, BLACK);
            
           for (int y = 0; y < dim; ++y)
           {
                for (int x = 0; x < dim; ++x)
                {
                    int index = y*dim + x;
                    int byte = index/8;
                    int bit = index%8;
                    bool value = (bitdeck[byte] & (1 << bit)) != 0;
                    bool isfilled = (filled[byte] & (1 << bit)) != 0;
                    bool isTested = (tested[byte] & (1 << bit)) != 0;
                    
                    if (value)
                    {
                        DrawRectangle(
                            x*rectSpacing + rectMargin, 
                            y*rectSpacing + rectMargin + topMargin, 
                            rectSize, 
                            rectSize, 
                            isTested ? YELLOW : (isfilled ? RED:DARKDARKBLUE));
                    }
                    else
                    {
                        DrawRectangleLines(
                            x*rectSpacing + rectMargin, 
                            y*rectSpacing + rectMargin + topMargin, 
                            rectSize, 
                            rectSize, 
                            DARKDARKBLUE);
                    }
                }
           }

        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    CloseWindow();        // Close window and OpenGL context
    //--------------------------------------------------------------------------------------

    return 0;
}


int Flood(int algo, const uint8* bitdeck, int dim, uint8* filled, int seedX, int seedY)
{
    switch (algo)
    {
        case 0: return Flood_1(bitdeck, dim, filled, seedX, seedY);
        case 1: return Flood_2(bitdeck, dim, filled, seedX, seedY);
        case 2: return Flood_3(bitdeck, dim, filled, seedX, seedY);
        default: return 0;
    }
}

int Flood_Incremental(int algo, const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, uint8* tested, int* numTested)
{
    switch (algo)
    {
        case 0: return Flood_1_Incremental(bitdeck, dim, filled, stack, stackCount, tested, numTested);
        case 1: return Flood_2_Incremental(bitdeck, dim, filled, stack, stackCount, tested, numTested);
        case 2: return Flood_3_Incremental(bitdeck, dim, filled, stack, stackCount, tested, numTested);
        default: return 0;
    }
}

int Flood_Incremental_Start(int algo, const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, int seedX, int seedY)
{
     switch (algo)
    {
        case 0: return Flood_1_Incremental_Start(bitdeck, dim, filled, stack, stackCount, seedX, seedY);
        case 1: return Flood_2_Incremental_Start(bitdeck, dim, filled, stack, stackCount, seedX, seedY);
        case 2: return Flood_3_Incremental_Start(bitdeck, dim, filled, stack, stackCount, seedX, seedY);
        default: return 0;
    }
}

// 4 directional test with stack

static inline int FillCell(const uint8* bitdeck, int dim, uint8* filled, int x, int y)
{
    // bitwise intentional to collapse to single branch
    if ((x < 0) | (x >= dim) | (y < 0) | (y >= dim)) return -1;
    
    int cell = y*dim + x;
    int byte = cell >> 3;
    uint8 bitmask = 1 << (cell&7);
    if (((bitdeck[byte] & bitmask) != 0) &&
        ((filled[byte] & bitmask) == 0))
    {
        filled[byte] |= bitmask;
        return cell;
    }
    return -1;
}

static inline int FillCellTest(const uint8* bitdeck, int dim, uint8* filled, uint8* tested, int* testCount, int x, int y)
{
    // bitwise intentional to collapse to single branch
    if ((x < 0) | (x >= dim) | (y < 0) | (y >= dim)) return -1;
    
    int cell = y*dim + x;
    int byte = cell >> 3;
    uint8 bitmask = 1 << (cell&7);
    tested[byte] |= bitmask;
    (*testCount)++;
    
    if (((bitdeck[byte] & bitmask) != 0) &&
        ((filled[byte] & bitmask) == 0))
    {
        filled[byte] |= bitmask;
        return cell;
    }
    return -1;
}

int Flood_1(const uint8* bitdeck, int dim, uint8* filled, int seedX, int seedY)
{
    int* stack = alloca(sizeof(int)*dim*dim); // Overkill for now
    int stackCount = 0;
    
    // fill seed cell push on stack
    int cellIndex = FillCell(bitdeck, dim, filled, seedX, seedY);
    if (cellIndex >= 0)
    {
        stack[stackCount++] = cellIndex;
    }
    
    int totalFilled = 0;
    
    // while something in stack
    while (stackCount)
    {
        ++totalFilled;
        
        // pop stack,
        cellIndex = stack[--stackCount];
        
        // for each of 4 directions, if bitdeck positive and not filled, fill cell, push on stack
        seedY = cellIndex/dim;
        seedX = cellIndex%dim;
        
        int left = FillCell(bitdeck, dim, filled, seedX, seedY-1);
        int right = FillCell(bitdeck, dim, filled, seedX, seedY+1);
        int up = FillCell(bitdeck, dim, filled, seedX-1, seedY);
        int down = FillCell(bitdeck, dim, filled, seedX+1, seedY);
        
        if (left >= 0) stack[stackCount++] = left;
        if (right >= 0) stack[stackCount++] = right;
        if (up >= 0) stack[stackCount++] = up;
        if (down >= 0) stack[stackCount++] = down;
    }
    
    return totalFilled;
}

int Flood_1_Incremental_Start(const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, int seedX, int seedY)
{
    IncrementalState* isStack = (IncrementalState*)stack;
    
    int cellIndex = FillCell(bitdeck, dim, filled, seedX, seedY);
    if (cellIndex >= 0)
    {
        IncrementalState newState;
        memset(&newState, 0, sizeof(newState));
        newState.Dfs.CellIndex = cellIndex;
       
        isStack[(*stackCount)++] = newState;
        return *stackCount;
    }
    return 0;
    
}

int Flood_1_Incremental(const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, uint8* tested, int* numTested)
{
     // while something in stack
    IncrementalState* isStack = (IncrementalState*)stack;
    int sc = *stackCount;
    int filledCount = 0;
    int tc = 0;
    
    if (sc)
    {
        DFSSIState* state = &isStack[sc-1].Dfs;
        
        int cellIndex = state->CellIndex;
        
        // for each of 4 directions, if bitdeck positive and not filled, fill cell, push on stack
        int y = cellIndex/dim;
        int x = cellIndex%dim;
        
        switch (state->Stage)
        {        
        case 0:
        {
            int top = FillCellTest(bitdeck, dim, filled, tested, &tc, x, y-1);
            if (top >= 0)
            {
                filledCount++;
                state->PushTop = true;
            }
            state->Stage++;
            break;
        }
        case 1:
        {
            int bottom = FillCellTest(bitdeck, dim, filled, tested, &tc, x, y+1);
            if (bottom >= 0)
            {
                filledCount++;
                state->PushBottom = true;
            }
            state->Stage++;
            break;
        }
        case 2:
        {
            int left = FillCellTest(bitdeck, dim, filled, tested, &tc, x-1, y);
            if (left >= 0)
            {
                filledCount++;
                state->PushLeft = true;
            }
            state->Stage++;
            break;
        }
        case 3:
        {
            int right = FillCellTest(bitdeck, dim, filled, tested, &tc, x+1, y);
            if (right >= 0)
            {
                filledCount++;
                state->PushRight = true;
            }
            state->Stage++;
           // Fallthrough
        }
        case 4:
        {
            DFSSIState oldState = *state;
            
            // Pop stack
            --sc;
            
            IncrementalState newState;
            memset(&newState, 0, sizeof(newState));
         
            if (oldState.PushTop)
            {
                newState.Dfs.CellIndex = (y-1)*dim + x;
                isStack[sc++] = newState;
            }
            if (oldState.PushBottom)
            {
                newState.Dfs.CellIndex = (y+1)*dim + x;
                isStack[sc++] = newState;
            }
            if (oldState.PushLeft)
            {
                newState.Dfs.CellIndex = y*dim + x-1;
                isStack[sc++] = newState;
            }
            if (oldState.PushRight)
            {
                newState.Dfs.CellIndex = y*dim + x+1;
                isStack[sc++] = newState;
            }
           
        }
        default:
            break;
        };

      
        *stackCount = sc;
    }
   
   *numTested = tc;
   return filledCount;
}

static inline int TestCell(const uint8* bitdeck, int dim, uint8* filled, int x, int y)
{
    // bitwise intentional to collapse to single branch
    if ((x < 0) | (x >= dim) | (y < 0) | (y >= dim)) return -1;
    
    int cell = y*dim + x;
    int byte = cell >> 3;
    uint8 bitmask = 1 << (cell&7);
    if (((bitdeck[byte] & bitmask) != 0) &&
        ((filled[byte] & bitmask) == 0))
    {
        return cell;
    }
    return -1;
}

static inline int TestCellTest(const uint8* bitdeck, int dim, uint8* filled, uint8* tested, int* testCount, int x, int y)
{
    // bitwise intentional to collapse to single branch
    if ((x < 0) | (x >= dim) | (y < 0) | (y >= dim)) return -1;
    
    int cell = y*dim + x;
    int byte = cell >> 3;
    uint8 bitmask = 1 << (cell&7);
    
    tested[byte] |= bitmask;
    (*testCount)++;
    
    if (((bitdeck[byte] & bitmask) != 0) &&
        ((filled[byte] & bitmask) == 0))
    {
        return cell;
    }
    return -1;
}

int Flood_2(const uint8* bitdeck, int dim, uint8* filled, int seedX, int seedY)
{
    int* stack = alloca(sizeof(int)*dim*dim); // This algo very stack efficient except in pathological worst case where up to dim*dim/2 could be required.
    int stackCount = 0;
    
    // Test and add seed cell to stack
    int cellIndex = seedY*dim + seedX;
    int byte = cellIndex >> 3;
    int bitmask = (1 << (cellIndex&7));
    
    if ((bitdeck[byte] & bitmask) != 0 &&
        (filled[byte] & bitmask) == 0)
    {
        stack[stackCount++] = cellIndex;
    }
    
    int numfilled = 0;
    
    // While something on stack
    while(stackCount)
    {    
        // pop cell index, 
        cellIndex = stack[--stackCount];
        
        int xleft, xright;
        
        // Span fill right
        int inc = 1;
        int y = cellIndex/dim;
        int x = cellIndex%dim;
       
        while (0 <= FillCell(bitdeck, dim, filled, x, y))
        {
            x+=inc;
            ++numfilled;
        }
        xright = x-inc;
       
        // span fill left
        x = (cellIndex%dim)-1;
        inc = -1;
        while (0 <= FillCell(bitdeck, dim, filled, x, y))
        {
            x+=inc;
             ++numfilled;
        }
        xleft = x-inc;
        
        // Scan above for seed, push
        if (y > 0)
        {
            x = xleft;
            int prevSeed = -1;
            while (x <= xright)
            {
                int newSeed = TestCell(bitdeck, dim, filled, x, y-1);
                if (newSeed >= 0 && prevSeed == -1)
                {
                    stack[stackCount++] = newSeed;
                }
                prevSeed = newSeed;
                ++x;
            }
        }
        
        // Scan below for seed, push
        if (y < dim-1)
        {
            x = xleft;
            int prevSeed = -1;
            while (x <= xright)
            {
                int newSeed = TestCell(bitdeck, dim, filled, x, y+1);
                if (newSeed >= 0 && prevSeed == -1)
                {
                    stack[stackCount++] = newSeed;
                }
                prevSeed = newSeed;
                ++x;
            }
        }
    }
    
    return numfilled;
}

int Flood_2_Incremental_Start(const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, int seedX, int seedY)
{
    IncrementalState* isStack = (IncrementalState*)stack;
    
    // Test and add seed cell to stack
    int cellIndex = seedY*dim + seedX;
    int byte = cellIndex >> 3;
    int bitmask = (1 << (cellIndex&7));
    
    if ((bitdeck[byte] & bitmask) != 0 &&
        (filled[byte] & bitmask) == 0)
    {
        IncrementalState newState;
        memset(&newState, 0, sizeof(newState));
        newState.Sf.CellIndex = cellIndex;
        
        isStack[(*stackCount)++] = newState;
    }
    
    return 0;
}

int Flood_2_Incremental(const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, uint8* tested, int* numTested)
{
    IncrementalState* isStack = (IncrementalState*)stack;
    int numfilled = 0;
    int tc = 0;
    
     // While something on stack
    int sc = *stackCount;
    if (sc)
    {    
        // pop cell index, 
        SpanFillSIState* state = &isStack[sc-1].Sf;
        int cellIndex = state->CellIndex;
        int y = cellIndex/dim;
        
        switch (state->Stage)
        {
        case 0:
        {
            // Initialize
            state->X = cellIndex%dim;
            state->Stage++;
            // fallthrough
        }
        case 1:
        {
            // Span fill right
            int inc = 1;
            if (0 <= FillCellTest(bitdeck, dim, filled, tested, &tc, state->X, y))
            {
                state->X+=inc;
                ++numfilled;
            }
            else
            {
                state->xRight = state->X - inc;
                state->X = (cellIndex%dim)-1;
                state->Stage++;
            }
            break;
        }
        case 2:
        {
            // span fill left
            int inc = -1;
            if (0 <= FillCellTest(bitdeck, dim, filled, tested, &tc, state->X, y))
            {
                state->X+=inc;
                 ++numfilled;
            }
            else
            {
                state->xLeft = state->X - inc;
                state->X = state->xLeft;
                state->PrevSeed = -1;
                state->Stage++;
            }
            break;
        }
        case 3:
        {
            // Scan above for seed, push
            if (y > 0)
            {
                if (state->X <= state->xRight)
                {
                    int newSeed = TestCellTest(bitdeck, dim, filled, tested, &tc, state->X, y-1);
                    if (newSeed >= 0 && state->PrevSeed == -1)
                    {
                        SFI_StackPush(newSeed);
                        state->PushCount++;
                    }
                    state->PrevSeed = newSeed;
                    state->X++;
                    break;
                }
               
            }
            
            state->X = state->xLeft;
            state->PrevSeed = -1;
            state->Stage++;
            // fallthrough;
            
        }
        case 4:
        {
            // Scan below for seed, push
            if (y < dim-1)
            {
                if (state->X <= state->xRight)
                {
                    int newSeed = TestCellTest(bitdeck, dim, filled, tested, &tc, state->X, y+1);
                    if (newSeed >= 0 && state->PrevSeed == -1)
                    {
                        SFI_StackPush(newSeed);
                        state->PushCount++;
                    }
                    state->PrevSeed = newSeed;
                    state->X++;
                    break;
                }
            }
            state->Stage++;
            // fallthrough
        }
        case 5:
        {
            // Resolve stack
            int pushCount = state->PushCount;
             --sc;
            while (pushCount)
            {
                IncrementalState newState;
                memset(&newState, 0, sizeof(newState));
                
                newState.Sf.CellIndex = SFI_StackPop();
                isStack[sc++] = newState;
                
                --pushCount;
            }
            
        }
        
        };
    }
    
    *numTested = tc;
    *stackCount = sc;
    return numfilled;
}


int Flood_3(const uint8* bitdeck, int dim, uint8* filled, int seedX, int seedY)
{
    // This algorithm is optimized for grids of 64 bits per line, but wider lines can be accomodated
    // by treating the overall grid as a grid of lines and adding cases for the horizontal neighbor tests.
    
    // Since we do full row operations, we cannot stack multiple discovered spans from the same row. We'll
    // stack an entire row, and there are only two directions we can look for new work in, up and down.
    // We'll prefer the down direction, so the stack size increases only when there is newly discovered
    // work in the upward direction that we leave behind as we push downward. Therefore, we can't possibly
    // push more than 'dim' rows to the stack, as that would imply we pushed more than one row per row
    // we traversed. In fact, since we must scan at least two cells horizontally to discover a new vertical
    // span, as we traverse the entire grid, we can't have pushed more than dim/2 rows to the stack, where
    // dim is the longest dimension. I've generated this test case in the provided worst-case fill pattern,
    // which will cause this algorithm to stack exactly 32 entries for a 64x64 grid if the fill is started
    // from either top corner.
    
    int* stack = alloca(sizeof(int)*((dim+1)/2)); 
    int stackCount = 0;
    int numFilled = 0;
    
    // Test and add seed cell to stack    
    int cellIndex = FillCell(bitdeck, dim, filled, seedX, seedY);
    if (cellIndex >= 0)
    {
        // We stack row numbers, not cell numbers
        stack[stackCount++] = cellIndex/64;
        ++numFilled;
    }
    
    uint64* bitRows = (uint64*)bitdeck;
    uint64* fillRows = (uint64*)filled;
    while (stackCount)
    {
        int rowIndex = stack[--stackCount];
        
        uint64 bitRow = bitRows[rowIndex];
        uint64 fillRow = fillRows[rowIndex];
        uint64 fillRowStart = fillRow;
        uint64 test = (fillRow<<1)&bitRow;
        uint64 fillRowPrev = 0;
        
        // Simulscan fill left
        while (test && (fillRowPrev != fillRow))
        {
            fillRowPrev = fillRow;
            fillRow |= test;
            test <<= 1;
            test &= bitRow;
        }
        
        // Simulscan fill right
        fillRowPrev = 0;
        test = (fillRow>>1)&bitRow;
        while (test && (fillRowPrev != fillRow))
        {
            fillRowPrev = fillRow;
            fillRow |= test;
            test >>= 1;
            test &= bitRow;
        }
        
        fillRows[rowIndex] = fillRow;
        numFilled += CountBits(fillRow ^ fillRowStart);
        
        // Bitfill up
        if (rowIndex > 0)
        {
            uint64 oldFill = fillRows[rowIndex-1];
            uint64 newFill = oldFill | (fillRow & bitRows[rowIndex-1]);
            if (oldFill != newFill)
            {
                fillRows[rowIndex-1] = newFill;
                stack[stackCount++] = rowIndex-1;
                numFilled += CountBits(oldFill ^ newFill);
            }
        }
        
        // Bitfill down
        if (rowIndex < dim-1)
        {
            uint64 oldFill = fillRows[rowIndex+1];
            uint64 newFill = oldFill | (fillRow & bitRows[rowIndex+1]);
            if (oldFill != newFill)
            {
                fillRows[rowIndex+1] = newFill;
                stack[stackCount++] = rowIndex+1;
                numFilled += CountBits(oldFill ^ newFill);
            }
        }
    }
    
    return numFilled;
}


int Flood_3_Incremental_Start(const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, int seedX, int seedY)
{
     // Fill seed cell and stack its row
     SimulSpanFillSIState* siStack = (SimulSpanFillSIState*)stack;
     
    int cellIndex = FillCell(bitdeck, dim, filled, seedX, seedY);
    if (cellIndex >= 0)
    {
        // We stack row numbers, not cell numbers
        SimulSpanFillSIState newState;
        memset(&newState, 0, sizeof(newState));
        
        newState.RowIndex = cellIndex/64;
        
        siStack[(*stackCount)++] = newState;
        return 1;
    }
    
    return 0;
}


int Flood_3_Incremental(const uint8* bitdeck, int dim, uint8* filled, int* stack, int* stackCount, uint8* tested, int* numTested)
{
    IncrementalState* siStack = (IncrementalState*)stack;
    uint64* bitRows = (uint64*)bitdeck;
    uint64* fillRows = (uint64*)filled;
    uint64* testRows = (uint64*)tested;
    
    int sc = *stackCount;
    int numFilled = 0;
    int testCount = 0;
    
    if (sc)
    {
        SimulSpanFillSIState* state = &siStack[sc-1].Ssf;
        int rowIndex = state->RowIndex;
        
        uint64 bitRow = bitRows[rowIndex];
        uint64 fillRow = fillRows[rowIndex];
            
        switch (state->Stage)
        {
            case 0:
            {
                state->FillRowPrev = 0;
                state->Test = (fillRow<<1)&bitRow;
                testCount++;
                
                state->Stage++;
                // Fallthrough
            }
            case 1:
            {                
                // Simulscan fill left
                if (state->Test)
                {
                    uint64 fillRowPrev = fillRow;
                    fillRow |= state->Test;
                    state->Test <<= 1;
                    state->Test &= bitRow;
                    
                    if (fillRowPrev != fillRow)
                    {
                        fillRows[rowIndex] = fillRow;
                        
                        uint64 newFilled = fillRow ^ fillRowPrev;
                        testRows[rowIndex] = newFilled;
                        numFilled += CountBits(newFilled);
                        break;
                    }
                }
                
                state->FillRowPrev = 0;
                state->Test = (fillRow>>1)&bitRow;
                
                state->Stage++;
                // Fallthrough
            }
            case 2:
            {
                // Simulscan fill right
                if (state->Test)
                {
                    uint64 fillRowPrev = fillRow;
                    fillRow |= state->Test;
                    state->Test >>= 1;
                    state->Test &= bitRow;
                    if (fillRowPrev != fillRow)
                    {
                        fillRows[rowIndex] = fillRow;
                        
                        uint64 newFilled = fillRow ^ fillRowPrev;
                        testRows[rowIndex] = newFilled;
                        numFilled += CountBits(newFilled);
                        break;
                    }
                }
                state->Stage++;
                // Fallthrough
            }
            case 3:
            {     
                // Bitfill up
                if (rowIndex > 0)
                {
                    testCount++;
                    testRows[rowIndex-1] |= fillRow;
                    
                    uint64 oldFill = fillRows[rowIndex-1];
                    uint64 newFill = oldFill | (fillRow & bitRows[rowIndex-1]);
                    if (oldFill != newFill)
                    {
                        fillRows[rowIndex-1] = newFill;
                        state->PushNextAbove = true;
                        
                        // Count how many we filled
                        numFilled += CountBits(oldFill ^ newFill);
                        state->Stage++;
                        break;
                    }

                }
                state->Stage++;
                // Fallthrough
            }
            case 4:
            {
            
                // Bitfill down
                if (rowIndex < dim-1)
                {
                    testCount++;
                    testRows[rowIndex+1] |= fillRow;
                    
                    uint64 oldFill = fillRows[rowIndex+1];
                    uint64 newFill = oldFill | (fillRow & bitRows[rowIndex+1]);
                    if (oldFill != newFill)
                    {
                        fillRows[rowIndex+1] = newFill;
                        state->PushNextBelow = true;
                        
                         // Count how many we filled
                        numFilled += CountBits(oldFill ^ newFill);
                    }
                }
                
                // Pop existing state, push new ones on
                
                // Can't read existing state after decrement and push of a new state.
                // Cache what we need.
                bool pushNextAbove = state->PushNextAbove;
                bool pushNextBelow = state->PushNextBelow;
                --sc;
                
                IncrementalState newState;
                memset(&newState, 0, sizeof(newState));
                if (pushNextAbove)
                {
                    newState.Ssf.RowIndex = rowIndex-1;
                    siStack[sc++] = newState;
                }
                if (pushNextBelow)
                {
                    newState.Ssf.RowIndex = rowIndex+1;
                    siStack[sc++] = newState;
                }
            }
            default:
                break;  
        }
    }
    
    *numTested = testCount;
    *stackCount = sc;
    
    return numFilled;
}

