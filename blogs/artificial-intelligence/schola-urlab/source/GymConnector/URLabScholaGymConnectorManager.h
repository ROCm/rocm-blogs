// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "TrainingUtils/GymConnectorManager.h"
#include "URLabScholaGymConnectorManager.generated.h"

/**
 * Drop-in replacement for AGymConnectorManager that defers Init() until the URLab AAMjManager
 * has finished compiling its mjModel.
 *
 * Why: Schola's stock manager calls `Connector->Init(Environments)` from BeginPlay, which in turn
 * calls `InitializeEnvironment` on every Schola env. URLab's AAMjManager also compiles its model
 * in BeginPlay. Unreal does not guarantee actor BeginPlay order, so on cold play the env may
 * cache joints/actuators from an empty articulation and ship empty obs/action spaces to the trainer.
 *
 * This subclass polls AAMjManager::IsInitialized() each tick (until success) and only then runs
 * the standard CollectEnvironments + Init flow. Use this actor in your level instead of the
 * stock AGymConnectorManager.
 */
UCLASS(meta = (DisplayName = "URLab Schola Gym Connector Manager"))
class YOURPROJECT_API AURLabScholaGymConnectorManager : public AGymConnectorManager
{
	GENERATED_BODY()

public:
	AURLabScholaGymConnectorManager();

	/** Optional cap on how long we wait for the URLab manager to compile (seconds). 0 = wait forever. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|URLab", meta = (ClampMin = "0.0"))
	float MaxWaitForManagerCompileSeconds = 30.f;

	virtual void Tick(float DeltaTime) override;

protected:
	virtual void BeginPlay() override;

private:
	bool TryInitNow();

	FTimerHandle PollTimer;
	double WaitStartTimeSeconds = 0.0;
	bool bInitDone = false;
};
