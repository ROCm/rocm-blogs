// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Copy into your game module's Private/ folder.

#include "URLabScholaGymConnectorManager.h"

#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "MuJoCo/Core/AMjManager.h"
#include "TimerManager.h"

DEFINE_LOG_CATEGORY_STATIC(LogURLabScholaGymMgr, Log, All);

AURLabScholaGymConnectorManager::AURLabScholaGymConnectorManager()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.bStartWithTickEnabled = true;
}

void AURLabScholaGymConnectorManager::BeginPlay()
{

	AActor::BeginPlay();

	if (!Connector)
	{
		UE_LOG(LogURLabScholaGymMgr, Warning, TEXT("No Connector assigned on '%s' — nothing to initialize."), *GetName());
		return;
	}

	WaitStartTimeSeconds = FPlatformTime::Seconds();

	if (TryInitNow())
	{
		return;
	}

	UE_LOG(LogURLabScholaGymMgr, Log, TEXT("Waiting for URLab manager to compile before initializing gym connector."));

	if (UWorld* World = GetWorld())
	{
		World->GetTimerManager().SetTimer(
			PollTimer,
			FTimerDelegate::CreateWeakLambda(this, [this]()
			{
				if (TryInitNow())
				{
					if (UWorld* W = GetWorld())
					{
						W->GetTimerManager().ClearTimer(PollTimer);
					}
				}
			}),
			0.05f,
			true);
	}
}

void AURLabScholaGymConnectorManager::Tick(float DeltaTime)
{
	if (!bInitDone)
	{
		return;
	}

	Super::Tick(DeltaTime);
}

bool AURLabScholaGymConnectorManager::TryInitNow()
{
	if (bInitDone)
	{
		return true;
	}

	AAMjManager* Mgr = AAMjManager::GetManager();
	const bool bManagerReady = Mgr && Mgr->IsInitialized();

	if (!bManagerReady)
	{
		if (MaxWaitForManagerCompileSeconds > 0.f)
		{
			const double Elapsed = FPlatformTime::Seconds() - WaitStartTimeSeconds;
			if (Elapsed >= MaxWaitForManagerCompileSeconds)
			{
				UE_LOG(LogURLabScholaGymMgr, Error,
					TEXT("Timed out after %.1fs waiting for the URLab manager to compile. Initializing the gym connector anyway — env spaces will likely be empty."),
					Elapsed);
			}
			else
			{
				return false;
			}
		}
		else
		{
			return false;
		}
	}

	if (Connector)
	{
		if (Mgr)
		{
			Mgr->SetPaused(true);
		}

		Connector->bRunEnvironmentsInParallel = false;

		TArray<TScriptInterface<IBaseScholaEnvironment>> Environments;
		Connector->CollectEnvironments(Environments);
		Connector->Init(Environments);

		UE_LOG(LogURLabScholaGymMgr, Log,
			TEXT("Gym connector initialized after %.3fs (URLab manager %s)."),
			FPlatformTime::Seconds() - WaitStartTimeSeconds,
			bManagerReady ? TEXT("ready") : TEXT("NOT ready — timed out"));
	}

	bInitDone = true;
	return true;
}
