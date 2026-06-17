// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Copy into your game module's Public/ folder.
// Replace YOURPROJECT_API with your module's API macro (for example MYGAME_API).

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Common/InteractionDefinition.h"
#include "Environment/CppOnlyMultiAgentEnvironmentInterface.h"
#include "Math/RandomStream.h"
#include "Spaces/DictSpace.h"
#include "StructUtils/InstancedStruct.h"
#include "TrainingDataTypes/AgentState.h"
#include "XArmEnvironment.generated.h"

class AActor;
class AMjArticulation;
class UMjActuator;
class UMjSensor;
class UMjJoint;
class AAMjManager;

/** One sensor reading for Blueprint-friendly snapshot data (UHT cannot reflect TMap<FString,TArray<float>>). */
USTRUCT(BlueprintType)
struct FURLabScholaNamedSensorReading
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Schola|URLab")
	FString SensorName;

	UPROPERTY(BlueprintReadOnly, Category = "Schola|URLab")
	TArray<float> Values;
};

/** One actuator command applied this step. */
USTRUCT(BlueprintType)
struct FURLabScholaNamedActuatorCommand
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Schola|URLab")
	FString ActuatorName;

	UPROPERTY(BlueprintReadOnly, Category = "Schola|URLab")
	float Value = 0.f;
};

/** Per-step context passed into reward / termination / info hooks (single logical agent). */
USTRUCT(BlueprintType)
struct FURLabScholaAgentSnapshot
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Schola|URLab")
	TObjectPtr<AMjArticulation> Robot = nullptr;

	UPROPERTY(BlueprintReadOnly, Category = "Schola|URLab")
	TArray<FURLabScholaNamedSensorReading> Observations;

	UPROPERTY(BlueprintReadOnly, Category = "Schola|URLab")
	TArray<FURLabScholaNamedActuatorCommand> ActionsApplied;

	UPROPERTY(BlueprintReadOnly, Category = "Schola|URLab")
	int32 GlobalStepIndex = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Schola|URLab")
	int32 EpisodeStepIndex = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Schola|URLab")
	int32 EnvironmentSeed = 0;
};

USTRUCT(BlueprintType)
struct FURLabScholaTermination
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|URLab")
	bool bTerminated = false;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|URLab")
	bool bTruncated = false;
};

/**
 * xArm6 reach-target Schola environment backed by one AMjArticulation.
 *
 * Observation (policy-facing, when bNormalizeObservations is true):
 *   - tcp_pos: TCP offset from target in UE cm, divided by ObsTcpScale (meters; ×100 at runtime).
 *   - Each arm joint: (pos / ObsJointPosScale, vel / ObsJointVelScale, acc / ObsJointAccScale).
 *
 * Reward / termination use the same UE-cm TCP readings as observations (distance via / 100.f for meter thresholds).
 *
 * Action: one scalar per arm actuator (six total; gripper excluded).
 */
UCLASS(Blueprintable, BlueprintType, meta = (DisplayName = "XArm Reach Environment"))
class YOURPROJECT_API AXArmEnvironment : public AActor, public ICppOnlyMultiAgentEnvironment
{
	GENERATED_BODY()

public:
	AXArmEnvironment();

	// ICppOnlyMultiAgentEnvironment
	virtual void InitializeEnvironment(TMap<FString, FInteractionDefinition>& OutAgentDefinitions) override;
	virtual void SeedEnvironment(int Seed) override;
	virtual void SetEnvironmentOptions(const TMap<FString, FString>& Options) override;
	virtual void Reset(TMap<FString, FInitialAgentState>& OutAgentState) override;
	virtual void Step(const TMap<FString, FInstancedStruct>& InActions, TMap<FString, FAgentState>& OutAgentStates) override;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|URLab")
	TObjectPtr<AMjArticulation> RobotArticulation;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|URLab")
	TObjectPtr<AAMjManager> RobotManager;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|URLab")
	int PhysicsStepsPerControlStep = 4;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|URLab")
	FString AgentId = TEXT("agent_0");

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|URLab|IO", meta = (ClampMin = "1.0"))
	float ObservationBound = 1.0e6f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|URLab")
	bool bForceUiControlSource = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Target")
	FVector TargetLocationMeters = FVector(0.5, 0.0, 0.2);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Target")
	bool bRandomizeTargetPerEpisode = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Target", meta = (EditCondition = "bRandomizeTargetPerEpisode"))
	FVector TargetRandomOffsetMaxMeters = FVector(0.2f, 0.4f, 0.2f);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Target")
	TObjectPtr<AActor> TargetActor;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|TCP")
	FString TcpSensorName = TEXT("tcp_pos");

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Reward")
	float DistanceScale = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Reward")
	float StepPenalty = -0.01f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Reward")
	float SuccessBonus = 5.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Reward", meta = (ClampMin = "0.0"))
	float CloseRegionMeters = 0.1f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Reward", meta = (ClampMin = "0.0"))
	float CloseRegionBonusScale = 0.2f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Reward", meta = (ClampMin = "0.0"))
	float SuccessThreshold = 0.1f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Reward", meta = (ClampMin = "0.0"))
	float SuccessMaxJointVelocity = 0.5f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Reward")
	bool bUseOutOfBounds = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Reward", meta = (ClampMin = "0.0"))
	float OutOfBoundsDistance = 1.5f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Reward")
	float OutOfBoundsPenalty = -5.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Reward")
	bool bTerminateOnOutOfBounds = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Reward", meta = (ClampMin = "0.0"))
	float ActionRatePenaltyScale = 0.005f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Reward", meta = (ClampMin = "0.0"))
	float JointVelocityPenaltyScale = 0.01f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Episode", meta = (ClampMin = "1"))
	int32 EpisodeStepLimit = 500;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Observation")
	bool bNormalizeObservations = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Observation", meta = (ClampMin = "0.01", EditCondition = "bNormalizeObservations"))
	float ObsTcpScale = 0.75f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Observation", meta = (ClampMin = "0.01", EditCondition = "bNormalizeObservations"))
	float ObsJointPosScale = PI;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Observation", meta = (ClampMin = "0.01", EditCondition = "bNormalizeObservations"))
	float ObsJointVelScale = 2.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Observation", meta = (ClampMin = "0.01", EditCondition = "bNormalizeObservations"))
	float ObsJointAccScale = 10.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "XArm|Observation", meta = (ClampMin = "0.01", EditCondition = "bNormalizeObservations"))
	float ObsClip = 3.0f;

protected:
	virtual void BeginPlay() override;
	virtual void OnConstruction(const FTransform& Transform) override;

#if WITH_EDITOR
	virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

	UFUNCTION(BlueprintNativeEvent, Category = "Schola|URLab")
	float ComputeReward(const FURLabScholaAgentSnapshot& Snapshot) const;
	virtual float ComputeReward_Implementation(const FURLabScholaAgentSnapshot& Snapshot) const;

	UFUNCTION(BlueprintNativeEvent, Category = "Schola|URLab")
	FURLabScholaTermination ComputeTermination(const FURLabScholaAgentSnapshot& Snapshot) const;
	virtual FURLabScholaTermination ComputeTermination_Implementation(const FURLabScholaAgentSnapshot& Snapshot) const;

	UFUNCTION(BlueprintNativeEvent, Category = "Schola|URLab")
	TMap<FString, FString> BuildInfo(const FURLabScholaAgentSnapshot& Snapshot) const;
	virtual TMap<FString, FString> BuildInfo_Implementation(const FURLabScholaAgentSnapshot& Snapshot) const;

private:
	void ApplyControlSourceIfNeeded();
	void RebuildComponentCaches();
	bool BuildDictSpaces(FDictSpace& OutObsSpace, FDictSpace& OutActionSpace) const;
	void ReadObservations(FDictPoint& OutDict) const;
	bool ApplyActionDict(const FInstancedStruct& ActionPoint, TMap<FString, float>& OutAppliedScalars);
	FURLabScholaAgentSnapshot MakeSnapshot(const FDictPoint& ObsDict,
		const TMap<FString, float>& ActionScalars) const;
	FURLabScholaAgentSnapshot BuildPolicyState(const TMap<FString, float>& ActionScalars,
		TInstancedStruct<FPoint>& OutObservations, TMap<FString, FString>& OutInfo) const;
	void ZeroActuatorControls();
	void SampleEpisodeTargetLocation();
	void UpdateTargetWorldCm();
	void SyncTargetActorToLocation();
	bool IsSuccessReached(float DistanceMeters, float JointVelocitySquaredSum) const;
	void NormalizePolicyObservations(TInstancedStruct<FPoint>& Observations) const;

	mutable TArray<float> PreviousActionVector;
	FVector EpisodeTargetLocationMeters = FVector(0.4, 0.0, 0.2);
	FVector TargetWorldCm = FVector::ZeroVector;
	FRandomStream TargetRandomStream;
	int32 TargetRandomSeed = 0;
	int32 TargetRandomEpisodeCounter = 0;

	int32 EnvironmentSeed = 0;
	TMap<FString, TObjectPtr<UMjSensor>> CachedSensors;
	TMap<FString, TObjectPtr<UMjJoint>> CachedJoints;
	TMap<FString, TObjectPtr<UMjActuator>> CachedActuators;

	bool bDefinitionsInitialized = false;
	int32 GlobalStepIndex = 0;
	int32 EpisodeStepIndex = 0;
};
